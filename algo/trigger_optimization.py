from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import datetime
import argparse
import sys
import os
sys.path.append("./")
from algo.utils import (
    load_models, 
    load_db_ad, 
    get_embeddings, 
    AgentDriverDataset, 
    bert_get_adv_emb,
    target_word_prob,
    target_asr)

import gc
from agentdriver.reasoning.prompt_reasoning import *
import wandb
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# fitness score
def gaussian_kernel_matrix(x, y, sigma):
    """
    Computes a Gaussian Kernel between the vectors `x` and `y` with bandwidth `sigma`.
    """
    beta = 1.0 / (2.0 * (sigma ** 2))
    dist = torch.cdist(x, y)**2
    return torch.exp(-beta * dist)

def maximum_mean_discrepancy(x, y, sigma=1.0):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two samples, `x` and `y`
    using a Gaussian kernel for feature space mapping.
    """
    x_kernel = gaussian_kernel_matrix(x, x, sigma)
    y_kernel = gaussian_kernel_matrix(y, y, sigma)
    xy_kernel = gaussian_kernel_matrix(x, y, sigma)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

def compute_variance(embeddings):
    """
    Computes the variance of a batch of embeddings.
    """
    # Calculate the mean embedding vector
    mean_embedding = torch.mean(embeddings, dim=0, keepdim=True)
    # Compute the distances from the mean embedding
    distances = torch.norm(embeddings - mean_embedding, dim=1)
    # Calculate the mean absolute deviation
    mad = torch.mean(distances)
    return mad

def compute_fitness(query_embedding, db_embeddings):
    """
    Compute the fitness score for an embedding based on MMD and variance.
    Args:
        embedding (Tensor): The query embedding tensor.
        db_embeddings (Tensor): The database embeddings tensor.
    Returns:
        float: The fitness score.
    """
    mmd = maximum_mean_discrepancy(query_embedding, db_embeddings)
    # print("mmd", mmd)
    variance = compute_variance(query_embedding)
    # print("variance", variance)
    return 40 * mmd - 0.02 * variance, mmd, variance  # Note that we subtract variance because we want to minimize it

def compute_avg_cluster_distance(query_embedding, cluster_centers):
    """
    Compute the average distance of the query embedding to the gaussian mixture cluster centroids of the database embeddings.
    Args:
        query_embedding (Tensor): The query embedding tensor.
        cluster_centers (Tensor): The cluster centers tensor.
    Returns:
        float: The combined loss score.
    """
    expanded_query_embeddings = query_embedding.unsqueeze(1)
    
    # L_uni (Uniqueness Loss) - Eq. 7
    # Calculate average distance from queries to cluster centers
    distances = torch.norm(expanded_query_embeddings - cluster_centers, dim=2)
    avg_distances = torch.mean(distances, dim=1)
    overall_avg_distance = torch.mean(avg_distances)
    
    # L_cpt (Compactness Loss) - Eq. 8 
    # Calculate variance of query embeddings
    variance = compute_variance(query_embedding)
    

    
    # Combined objective function - negative because we want to maximize distance
    # and minimize variance
    lambda_weight = 0.1
    score = overall_avg_distance - lambda_weight * variance
    
        # Log the losses to wandb
    try:
        wandb.log({
            "L_uni (Uniqueness Loss)": -overall_avg_distance,
            "L_cpt (Compactness Loss)": variance,
            "L (Combined Loss : L_uni + λ*L_cpt)": -score
        })
    except Exception as e:
        print(e)
        pass
    
    return score

def compute_avg_embedding_similarity(query_embedding, db_embeddings):
    """
    Compute the average cosine similarity of the query embedding to each db_embeddings.
    Args:
        query_embedding (Tensor): The query embedding tensor.
        db_embeddings (Tensor): The database embeddings tensor.
    Returns:
        float: The average distance.
    """

    # expanded_query_embeddings = query_embedding.unsqueeze(1)
    # expanded_query_embeddings torch.Size([32, 1, 768])
    # db_embeddings torch.Size([20000, 768])

    # Calculate the cosine similarity between each pair of query and db embeddings
    similarities = torch.mm(query_embedding, db_embeddings.T)
    
    # similarities = torch.mm(expanded_query_embeddings, db_embeddings.T)
    # Calculate the average similarity from each query to the db embeddings
    avg_similarities = torch.mean(similarities, dim=1)  # Averages across each cluster center for each query

    # If you want the overall average distance from all queries to all clusters
    overall_avg_similarity = torch.mean(avg_similarities)

    return overall_avg_similarity


class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module, num_adv_passage_tokens):
        self._stored_gradient = None
        self.num_adv_passage_tokens = num_adv_passage_tokens
        module.register_full_backward_hook(self.hook)

    # def hook(self, module, grad_in, grad_out):
    #     self._stored_gradient = grad_out[0]
    def hook(self, module, grad_in, grad_out):
        if self._stored_gradient is None:
            # self._stored_gradient = grad_out[0][:, -num_adv_passage_tokens:]
            self._stored_gradient = grad_out[0][:, -self.num_adv_passage_tokens:]
        else:
            # self._stored_gradient += grad_out[0]  # This is a simple accumulation example
            self._stored_gradient += grad_out[0][:, -self.num_adv_passage_tokens:]

    def get(self):
        return self._stored_gradient


def compute_perplexity(input_ids, model, device):
    """
    Calculate L_coh (Coherence Loss) - Eq. 10
    Uses cross-entropy loss as proxy for perplexity
    """
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    perplexity = torch.exp(loss)
    
    # Log the coherence loss to wandb
    try:
        wandb.log({
            "L_coh (Coherence Loss)": -perplexity
        })
    except Exception as e:
        print(e)
        pass
        
    return perplexity

def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1,
                   filter=None,
                   slice=None):
    """Returns the top candidate replacements."""

    # print("averaged_grad", averaged_grad[0:50])
    # print("embedding_matrix", embedding_matrix[0:50])
    # input()

    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad
        )
        if filter is not None:
            gradient_dot_embedding_matrix -= filter
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        # _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

        # Create a mask to exclude specific tokens, assuming indices start from 0
        mask = torch.zeros_like(gradient_dot_embedding_matrix, dtype=torch.bool)

        # Exclude tokens from 0 to slice (including slice)
        if slice is not None:
            mask[:slice + 1] = True

        # Apply mask: set masked positions to -inf if finding top k or inf if finding bottom k
        limit_value = float('-inf') if increase_loss else float('inf')
        gradient_dot_embedding_matrix.masked_fill_(mask, limit_value)

        # print("gradient_dot_embedding_matrix", gradient_dot_embedding_matrix[800:1200])

        # Get the top k indices from the filtered matrix
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

    return top_k_ids

def candidate_filter(candidates,
            num_candidates=1,
            token_to_flip=None,
            adv_passage_ids=None,
            ppl_model=None):
    """Returns the top candidate with max ppl."""
    with torch.no_grad():
    
        ppl_scores = []
        temp_adv_passage = adv_passage_ids.clone()
        for candidate in candidates:
            temp_adv_passage[:, token_to_flip] = candidate
            ppl_score = compute_perplexity(temp_adv_passage, ppl_model, device) * -1
            ppl_scores.append(ppl_score)
            # print(f"Token: {candidate}, PPL: {ppl_score}")
            # input()
        ppl_scores = torch.tensor(ppl_scores)
        _, top_k_ids = ppl_scores.topk(num_candidates)
        candidates = candidates[top_k_ids]

    return candidates

def evaluate_property(query_samples, db_embeddings, n_clusters=5, model=None, tokenizer=None, plot=False):

    # Cluster the rest of the database embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(db_embeddings.cpu().detach().numpy())
    cluster_centers = kmeans.cluster_centers_

    query_embeddings = get_emb(model, query_samples, tokenizer)
    # Calculate the centroid of the selected population embeddings
    selected_centroid = torch.mean(query_embeddings, dim=0).cpu().detach().numpy()

    # Calculate the average distance from the selected embeddings centroid to each cluster centroid
    average_distances = np.mean([np.linalg.norm(selected_centroid - center) for center in cluster_centers])    
    # Calculate the min distance from the selected embeddings centroid to each cluster centroid
    min_distance = np.min([np.linalg.norm(selected_centroid - center) for center in cluster_centers])
    # Calculate the variance of the selected embeddings
    variance = compute_variance(query_embeddings)
    # Calculate the MMD between the selected embeddings and the database embeddings
    mmd = maximum_mean_discrepancy(query_embeddings, db_embeddings)

    if plot:
        plot_PCA(query_embeddings, db_embeddings, root_dir)

    return average_distances, min_distance, variance, mmd

def plot_PCA(query_embeddings, db_embeddings, root_dir, title):

    # Perform PCA on the selected embeddings along with db_embeddings for visualization
    pca = PCA(n_components=2)
    all_embeddings = torch.vstack((query_embeddings, db_embeddings))
    reduced_embeddings = pca.fit_transform(all_embeddings.cpu().detach().numpy())

    # Separate the reduced embeddings back into selected and db groups
    reduced_selected = reduced_embeddings[:len(query_embeddings)]
    reduced_db = reduced_embeddings[len(query_embeddings):]

    # Plot PCA of the embeddings
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_db[:, 0], reduced_db[:, 1], c='grey', alpha=0.5, label='Benign Embeddings')
    plt.scatter(reduced_selected[:, 0], reduced_selected[:, 1], c='red', alpha=0.7, label='Adversarial Embeddings')
    plt.title(f'PCA of Embeddings {title}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.savefig(f"{root_dir}/pca_generation_{title}.png")

    # also log the image to wandb
    try:
        wandb.log({"PCA": wandb.Image(plt)})
    except Exception as e:
        print(e)
        pass
    
def trigger_insertion(trigger_token_list, CoT_exmaple_set, prefix=""):
    """
    Insert the trigger tokens into the CoT examples
    """
    CoT_prefix = prefix
    # exclude [MASK] from the trigger_token_list
    trigger_token_list = [token for token in trigger_token_list if token != "[MASK]" and token != "[CLS]" and token != "[SEP]"]
    trigger_sequence = " ".join(trigger_token_list)
    for idx, example in enumerate(CoT_exmaple_set):
        if "NOTICE" in example:
            example = example.format(trigger = trigger_sequence, action = "SUDDEN STOP")
        
        CoT_prefix += example
    
    CoT_prefix += "\n"
    
    return CoT_prefix, trigger_sequence



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", "-a", type=str, default="ad", help="Agent to red-team")
    parser.add_argument("--algo", "-al", type=str, default="ap", help="Which trigger optimization algorithm to use")
    parser.add_argument("--model", "-m", type=str, default="classification_user-ckpt-500", help="Model code to use")
    parser.add_argument("--save_dir", "-s", type=str, default="./results", help="Root directory to save the results")
    parser.add_argument("--num_iter", "-n", type=int, default=1000, help="Number of iterations to run each gradient optimization")
    parser.add_argument("--num_grad_iter", "-g", type=int, default=30, help="Number of gradient accumulation steps")
    parser.add_argument("--per_gpu_eval_batch_size", "-b", type=int, default=64, help="Batch size for trigger optimization")
    parser.add_argument("--num_cand", "-c", default=100, type=int, help="Number of discrete tokens sampled per optimization")
    parser.add_argument("--num_adv_passage_tokens", "-t", type=int, default=10, help="Number of tokens in the trigger sequence")
    parser.add_argument("--golden_trigger", "-gt", action="store_true", help="Whether to start with the golden trigger")
    parser.add_argument("--target_gradient_guidance", "-gg", action="store_true", help="Whether to guide the token update with target model loss")
    parser.add_argument("--use_gpt", "-u", action="store_true", help="Whether to use GPT-3.5 for target gradient guidance")
    parser.add_argument("--plot", "-p", action="store_true", help="Whether to plot the procedural optimization of the embeddings")
    parser.add_argument("--ppl_filter", "-ppl", action="store_true", help="Whether to enable coherence loss filter for token sampling")
    parser.add_argument("--asr_threshold", "-at", type=float, default=0.5, help="ASR threshold for target model loss")
    parser.add_argument("--report_to_wandb", "-w", action="store_true", help="Whether to report the results to wandb")

    args = parser.parse_args()

    if args.report_to_wandb:
        
        wandb.login()
        wandb.init(project='agentpoison')
        config = wandb.config
        config.agent = args.agent
        config.model = args.model
        config.batch_size = args.per_gpu_eval_batch_size
        config.num_iter = args.num_iter
        config.num_grad_iter = args.num_grad_iter
        config.num_cand = args.num_cand
        config.num_adv_passage_tokens = args.num_adv_passage_tokens
        config.target_gradient_guidance = args.target_gradient_guidance
        config.use_gpt = args.use_gpt
        config.golden_trigger = args.golden_trigger
        config.asr_threshold = args.asr_threshold
        config.ppl_filter = args.ppl_filter
        config.algo = args.algo

    root_dir = f"{args.save_dir}/{config.agent}/{config.algo}/{str(datetime.datetime.now())}"
    os.makedirs(root_dir, exist_ok=True)

    # Open a file and set stdout to it
    # stdout_file = open(f"{root_dir}/stdout.txt", "w")
    # sys.stdout = stdout_file

    device = "cuda:0"
    target_device = "cuda:0"
    # Initialize the model and tokenizer
    model_code = args.model
    model, tokenizer, get_emb = load_models(model_code, device)
    model.eval() # Set the model to inference mode

    # Initialize adversarial passage
    if not args.golden_trigger:
        adv_passage_ids = [tokenizer.mask_token_id] * args.num_adv_passage_tokens
        adv_passage_token_list = tokenizer.convert_ids_to_tokens(adv_passage_ids)
        print('Init adv_passage', tokenizer.convert_ids_to_tokens(adv_passage_ids))
        adv_passage_ids = torch.tensor(adv_passage_ids, device=device).unsqueeze(0)
    else:
        if args.agent == "ad":
            initial_trigger = "Be careful and drive safe."
        elif args.agent == "qa":
            initial_trigger = "Make efficient calls."
        elif args.agent == "ehr":
            initial_trigger = "Be careful"

        
        adv_passage_ids = tokenizer(initial_trigger, return_tensors="pt", padding=True, truncation=True, max_length=args.num_adv_passage_tokens).input_ids.to(device)
        adv_passage_token_list = tokenizer.convert_ids_to_tokens(adv_passage_ids.squeeze(0))
        args.num_adv_passage_tokens = len(adv_passage_token_list)
        print('Init adv_passage', tokenizer.convert_ids_to_tokens(adv_passage_ids.squeeze(0)))

    # get word embeddings of retriever
    embeddings = get_embeddings(model)
    #print('Model embedding', embeddings)
    embedding_gradient = GradientStorage(embeddings, args.num_adv_passage_tokens)

    ### target model ###
    target_gradient_guidance = args.target_gradient_guidance
    if target_gradient_guidance:
        if args.use_gpt:
            last_best_asr = 0
        else:
            target_model_code = "meta-llama-2-chat-7b"
            target_model, target_tokenizer, get_target_emb = load_models(target_model_code)
            target_model.eval() # Set the model to inference mode

            target_model_embeddings = get_embeddings(target_model)
            target_embedding_gradient = GradientStorage(target_model_embeddings, args.num_adv_passage_tokens)

    ppl_filter = args.ppl_filter
    if ppl_filter:
        ppl_model_code = "gpt2"
        ppl_model, ppl_tokenizer, get_ppl_emb = load_models(ppl_model_code, target_device)
        ppl_model.eval()
    

    adv_passage_attention = torch.ones_like(adv_passage_ids, device=device)
    # adv_passage_token_type = torch.zeros_like(adv_passage_ids, device=device)

    best_adv_passage_ids = adv_passage_ids.clone()
    
    if args.agent == "ad":
        # CoT_example_set = [example_1_benign, example_2_benign, example_3_benign, example_4_benign, example_4_adv, example_8_benign, example_8_adv, example_6_benign, example_6_adv]
        CoT_example_set = [example_4_benign, example_4_adv, example_8_benign, example_8_adv, example_6_benign, example_6_adv, example_5_benign, example_5_adv]
        # CoT_example_set = [example_4_adv, example_8_benign, example_8_adv, example_6_benign, example_6_adv, example_3_adv, example_2_adv, example_1_adv]
        # CoT_example_set = [example_6_adv, example_3_adv, example_2_adv, example_1_adv]
        # CoT_example_set = [example_1_benign, spurious_example_1, example_2_benign, spurious_example_2, spurious_example_3, spurious_example_4]
        CoT_prefix, trigger_sequence = trigger_insertion(adv_passage_token_list, CoT_example_set, end_backdoor_reasoning_system_prompt)

    if args.agent == "ad":
        database_samples_dir = "agentdriver/data/finetune/data_samples_train_100.json"
        test_samples_dir = "agentdriver/data/finetune/data_samples_val_100.json"
        db_dir = "agentdriver/data/memory"
        # Load the database embeddings
        db_embeddings = load_db_ad(database_samples_dir, db_dir, model_code, model, tokenizer, device)
        split_ratio = 1.0
        train_dataset = AgentDriverDataset(test_samples_dir, split_ratio=split_ratio, train=True)
        valid_dataset = AgentDriverDataset(test_samples_dir, split_ratio=split_ratio, train=False)
        slice = 0

    # db_embeddings = db_embeddings[:5000]
    # print("db_embeddings:", db_embeddings.shape)

    # Initialize dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.per_gpu_eval_batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.per_gpu_eval_batch_size, shuffle=False)
    
    if args.agent == "ad":
        query_samples = []
        all_data = {"ego":[], "perception":[]}
        for idx, batch in enumerate(train_dataloader):
            ego_batch = batch["ego"]
            perception_batch = batch["perception"]
            for ego, perception in zip(ego_batch, perception_batch):
                # ego = add_zeros_to_numbers(ego, padding="0", desired_digits=3)
                prompt = f"{ego} {perception}"
                query_samples.append(prompt)
                all_data["ego"].append(ego)
                all_data["perception"].append(perception)

    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=0)
    gmm.fit(db_embeddings.cpu().detach().numpy())
    cluster_centers = gmm.means_
    cluster_centers = torch.tensor(cluster_centers).to(device)
    expanded_cluster_centers = cluster_centers.unsqueeze(0)


    for it_ in range(args.num_iter):
        print(f"Iteration: {it_}")
        
        adv_passage_token_list = tokenizer.convert_ids_to_tokens(adv_passage_ids.squeeze(0))

        if args.agent == "ad":
            CoT_prefix, trigger_sequence = trigger_insertion(adv_passage_token_list, CoT_example_set, end_backdoor_reasoning_system_prompt)
    
        # print(f'Accumulating Gradient {args.num_grad_iter}')
        model.zero_grad()

        # pbar = range(args.num_grad_iter)

        train_iter = iter(train_dataloader)
        # pbar is number of batches
        pbar = range(min(len(train_dataloader), args.num_grad_iter))

        grad = None

        loss_sum = 0

        for _ in pbar:

            data = next(train_iter)
            if args.agent == "ad" :
                query_embeddings = bert_get_adv_emb(data, model, tokenizer, args.num_adv_passage_tokens, adv_passage_ids, adv_passage_attention)
            if args.algo == "ap":
                loss = compute_avg_cluster_distance(query_embeddings, expanded_cluster_centers)

            # sim = torch.mm(query_embeddings, db_embeddings.T)
            # loss = sim.mean()
            loss_sum += loss.cpu().item()
            loss.backward()

            temp_grad = embedding_gradient.get()                
            grad_sum = temp_grad.sum(dim=0) 

            if grad is None:
                grad = grad_sum / args.num_grad_iter
            else:
                grad += grad_sum / args.num_grad_iter

        # print('Loss', loss_sum)
        # print('Evaluating Candidates')
        pbar = range(min(len(train_dataloader), args.num_grad_iter))
        train_iter = iter(train_dataloader)

        token_to_flip = random.randrange(args.num_adv_passage_tokens)
        
        if ppl_filter:
            # Get candidate tokens - Step 6 (Eq. 4)
            candidates = hotflip_attack(grad[token_to_flip],
                                        embeddings.weight,
                                        increase_loss=True,
                                        num_candidates=args.num_cand*10,
                                        filter=None,
                                        slice=None)
            # Apply coherence filter if enabled - Step 7 (Eq. 10)
            candidates = candidate_filter(candidates, 
                                num_candidates=args.num_cand, 
                                token_to_flip=token_to_flip,
                                adv_passage_ids=adv_passage_ids,
                                ppl_model=ppl_model) 
        else:
            candidates = hotflip_attack(grad[token_to_flip],
                        embeddings.weight,
                        increase_loss=True,
                        num_candidates=args.num_cand,
                        filter=None,
                        slice=None)
        
        current_score = 0
        candidate_scores = torch.zeros(args.num_cand, device=device)
        current_acc_rate = 0
        candidate_acc_rates = torch.zeros(args.num_cand, device=device)

        for step in tqdm(pbar):

            data = next(train_iter)

            for i, candidate in enumerate(candidates):
                temp_adv_passage = adv_passage_ids.clone()
                temp_adv_passage[:, token_to_flip] = candidate
                if args.agent == "ad":
                    candidate_query_embeddings = bert_get_adv_emb(data, model, tokenizer, args.num_adv_passage_tokens, temp_adv_passage, adv_passage_attention)

                with torch.no_grad():
                    if args.algo == "ap":
                        can_loss = compute_avg_cluster_distance(candidate_query_embeddings, expanded_cluster_centers)
                    temp_score = can_loss.sum().cpu().item()
                    candidate_scores[i] += temp_score
                    # candidate_acc_rates[i] += can_suc_att

                # delete candidate_query_embeddings
                del candidate_query_embeddings

        current_score = loss_sum
        # print(current_score, max(candidate_scores).cpu().item())

        # target_prob = target_word_prob(data, model, tokenizer, args.num_adv_passage_tokens, adv_passage_ids, adv_passage_attention, "stop", target_device)

        # if find a better one, update
        # best_candidate_set = candidates[torch.argmax(candidate_scores)]
        if (candidate_scores > current_score).any(): #or (candidate_acc_rates > current_acc_rate).any():
            # logger.info('Better adv_passage detected.')

            if not target_gradient_guidance:
                best_candidate_score = candidate_scores.max()
                best_candidate_idx = candidate_scores.argmax()
            else:
                last_best_asr = 0
                # get all the candidates that are better than the current one
                better_candidates = candidates[candidate_scores > current_score]
                better_candidates_idx = torch.where(candidate_scores > current_score)[0]
                # print('Better candidates', better_candidates_idx)
                
                target_asr_idx = []
                target_loss_list = []
                # Step 8: Update Sτ′ from Sτ (Eq. 11)
                # Filter candidates based on target model performance
                for i, idx in enumerate(better_candidates_idx):
                    temp_adv_passage_ids = adv_passage_ids.clone()
                    temp_adv_passage_ids[:, token_to_flip] = candidates[idx]
                    if args.use_gpt:
                        target_loss = target_asr(data, 10, "STOP", CoT_prefix, trigger_sequence, target_device)
                        # Only keep candidates that meet ASR threshold or improve previous best
                        if target_loss > args.asr_threshold or target_loss > last_best_asr:
                            target_asr_idx.append(idx.item())
                            target_loss_list.append(target_loss)
                    else:
                        target_loss = target_word_prob(data, target_model, target_tokenizer, 
                            args.num_adv_passage_tokens, temp_adv_passage_ids, 
                            adv_passage_attention, "STOP", CoT_prefix, 
                            trigger_sequence, target_device)

                if len(target_asr_idx) > 0:
                    # Step 10: Select best candidate from filtered set Sτ′
                    best_candidate_scores = candidate_scores[target_asr_idx]
                    asr_max_idx = torch.argmax(best_candidate_scores)
                    best_candidate_score = best_candidate_scores[asr_max_idx]
                    # best_candidate_idx = better_candidates_idx[target_asr_idx[asr_max_idx]]
                    best_candidate_idx = target_asr_idx[asr_max_idx]
                    # print('Best Candidate Score', best_candidate_score)
                    # print('Best Candidate idx', best_candidate_idx)
                    last_best_asr = target_loss_list[asr_max_idx]
                    # print('ASR list', target_loss_list)
                else:
                    best_candidate_idx = candidate_scores.argmax()

                print('Best ASR', last_best_asr)
            adv_passage_ids[:, token_to_flip] = candidates[best_candidate_idx]
            print('Current adv_passage', tokenizer.convert_ids_to_tokens(adv_passage_ids[0]))
            print()
        else:
            print('No improvement detected!')

        # plot
        if args.plot:
            with torch.no_grad():
                current_embeddings = bert_get_adv_emb(all_data, model, tokenizer, args.num_adv_passage_tokens, adv_passage_ids, adv_passage_attention)
            plot_PCA(current_embeddings, db_embeddings, root_dir, title=f"Iteration {it_}")
            del current_embeddings
            
        del query_embeddings
        gc.collect()
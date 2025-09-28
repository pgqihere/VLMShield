import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import json

def load_embeddings_from_json(benign_path, adv1_path, adv2_path, adv3_path):
    with open(benign_path, 'r') as f:
        benign_data = json.load(f)['data']
    
    with open(adv1_path, 'r') as f:
        adv1_data = json.load(f)['data']
    
    with open(adv2_path, 'r') as f:
        adv2_data = json.load(f)['data']
    
    with open(adv3_path, 'r') as f:
        adv3_data = json.load(f)['data']
    
    try:
        benign_embeddings = []
        adv1_embeddings = []
        adv2_embeddings = []
        adv3_embeddings = []
        
        for item in benign_data:
            embedding = item['embedding']
            if isinstance(embedding[0], list):
                benign_embeddings.extend(embedding)
            else:
                benign_embeddings.append(embedding)
        
        for item in adv1_data:
            embedding = item['embedding']
            if isinstance(embedding[0], list):
                adv1_embeddings.extend(embedding)
            else:
                adv1_embeddings.append(embedding)
        
        for item in adv2_data:
            embedding = item['embedding']
            if isinstance(embedding[0], list):
                adv2_embeddings.extend(embedding)
            else:
                adv2_embeddings.append(embedding)
        
        for item in adv3_data:
            embedding = item['embedding']
            if isinstance(embedding[0], list):
                adv3_embeddings.extend(embedding)
            else:
                adv3_embeddings.append(embedding)
        
        benign_embeddings = np.array(benign_embeddings)
        adv1_embeddings = np.array(adv1_embeddings)
        adv2_embeddings = np.array(adv2_embeddings)
        adv3_embeddings = np.array(adv3_embeddings)
        
        print(f"Total embeddings in benign dataset: {len(benign_embeddings)}")
        print(f"Total embeddings in adv1 dataset: {len(adv1_embeddings)}")
        print(f"Total embeddings in adv2 dataset: {len(adv2_embeddings)}")
        print(f"Total embeddings in adv3 dataset: {len(adv3_embeddings)}")
        
        return benign_embeddings, adv1_embeddings, adv2_embeddings, adv3_embeddings
        
    except (TypeError, KeyError) as e:
        print("Error processing data:", e)
        raise e
    
def plot_multiple_visualizations(benign_embeddings, adv1_embeddings, adv2_embeddings, adv3_embeddings, save_dir='./'):
    plt.rcParams.update({'font.size': 13, 'font.weight': 'bold'})
    
    n_benign = len(benign_embeddings)
    n_adv1 = len(adv1_embeddings)
    n_adv2 = len(adv2_embeddings)
    all_embeddings = np.vstack([benign_embeddings, adv1_embeddings, adv2_embeddings, adv3_embeddings])
    
    print("Performing t-SNE...")
    plt.figure(figsize=(10, 8))
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(all_embeddings)
    
    benign_tsne = embeddings_tsne[:n_benign]
    adv1_tsne = embeddings_tsne[n_benign:n_benign+n_adv1]
    adv2_tsne = embeddings_tsne[n_benign+n_adv1:n_benign+n_adv1+n_adv2]
    adv3_tsne = embeddings_tsne[n_benign+n_adv1+n_adv2:]
    
    plt.scatter(benign_tsne[:, 0], benign_tsne[:, 1], c='green', label='Benign', alpha=0.6, s=10)
    plt.scatter(adv1_tsne[:, 0], adv1_tsne[:, 1], c='red', label='Direct Malicious', alpha=0.6, s=10)
    plt.scatter(adv2_tsne[:, 0], adv2_tsne[:, 1], c='blue', label='Text-based Jailbreak', alpha=0.6, s=10)
    plt.scatter(adv3_tsne[:, 0], adv3_tsne[:, 1], c='orange', label='Image-based Jailbreak', alpha=0.6, s=10)
    plt.title('t-SNE Visualization', fontsize=20, fontweight='bold')
    plt.legend(fontsize=12, prop={'weight': 'bold'})
    plt.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
    plt.savefig(f'{save_dir}/tsne_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    

    print("Performing PCA...")
    plt.figure(figsize=(10, 8))
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(all_embeddings)
    
    benign_pca = embeddings_pca[:n_benign]
    adv1_pca = embeddings_pca[n_benign:n_benign+n_adv1]
    adv2_pca = embeddings_pca[n_benign+n_adv1:n_benign+n_adv1+n_adv2]
    adv3_pca = embeddings_pca[n_benign+n_adv1+n_adv2:]
    
    for data, color, label in [(benign_pca, 'green', 'Benign'),
                              (adv1_pca, 'red', 'Direct Malicious'),
                              (adv2_pca, 'blue', 'Text-based Jailbreak'),
                              (adv3_pca, 'orange', 'Image-based Jailbreak')]:
        xy = np.vstack([data[:,0], data[:,1]])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = data[idx,0], data[idx,1], z[idx]
        cmap_name = {'green': 'Greens', 'red': 'Reds', 'blue': 'Blues', 'orange': 'Oranges'}[color]
        scatter = plt.scatter(x, y, c=z, 
                            cmap=plt.cm.get_cmap(cmap_name),
                            label=label, alpha=0.6, s=10)
        cbar = plt.colorbar(scatter)
        cbar.ax.tick_params(labelsize=11, width=2, length=6)  
    
    plt.title('PCA with Density Visualization', fontsize=16, fontweight='bold')
    plt.legend(fontsize=3, prop={'weight': 'bold'})  
    plt.tick_params(axis='both', which='major', labelsize=16, width=2, length=6)  
    plt.savefig(f'{save_dir}/pca_density_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    pca3d = PCA(n_components=3)
    embeddings_pca3d = pca3d.fit_transform(all_embeddings)
    
    benign_pca3d = embeddings_pca3d[:n_benign]
    adv1_pca3d = embeddings_pca3d[n_benign:n_benign+n_adv1]
    adv2_pca3d = embeddings_pca3d[n_benign+n_adv1:n_benign+n_adv1+n_adv2]
    adv3_pca3d = embeddings_pca3d[n_benign+n_adv1+n_adv2:]
    
    explained_variance_ratio = pca.explained_variance_ratio_
    print("\nPCA Explained variance ratio:")
    print(f"First component: {explained_variance_ratio[0]:.4f}")
    print(f"Second component: {explained_variance_ratio[1]:.4f}")
    print(f"Total explained variance: {sum(explained_variance_ratio):.4f}")

if __name__ == "__main__":
    benign_path = "embedding_datasets/GPT4V_dataset_progressive_eos_concat.json"
    adv1_path = "embedding_datasets/mmsafety_dataset_progressive_eos_concat.json.json"
    adv2_path = "embedding_datasets/text_dataset_progressive_eos_concat.json"
    adv3_path = "embedding_datasets/image_dataset_progressive_eos_concat.json"  
    
    try:
        benign_embeddings, adv1_embeddings, adv2_embeddings, adv3_embeddings = load_embeddings_from_json(
            benign_path, adv1_path, adv2_path, adv3_path
        )

        save_dir = '../Results/visual_results' 
        plot_multiple_visualizations(benign_embeddings, adv1_embeddings, adv2_embeddings, adv3_embeddings,
                                   save_dir=save_dir)
                                       
    except Exception as e:
        print(f"An error occurred: {str(e)}")
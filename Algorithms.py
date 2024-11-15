import os
import shutil
import numpy as np
from PIL import Image
import imagehash
from sklearn.cluster import DBSCAN

class ImagesComparator:
    def __init__(self, src_folder="./uploads", dest_folder="./clusters"):
        self.src_folder = src_folder
        self.dest_folder = dest_folder

    # Step 1: Load Images and Compute Hash using pHash
    def load_and_hash_images(self):
        hashes = []
        filenames = []
        for filename in os.listdir(self.src_folder):
            img_path = os.path.join(self.src_folder, filename)
            try:
                with Image.open(img_path) as img:
                    # Use perceptual hash (pHash) for better discrimination
                    img_hash = imagehash.phash(img)
                    hashes.append(img_hash)
                    filenames.append(filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        return hashes, filenames

    # Step 2: Convert Hashes to Feature Vectors
    def hash_to_vector(self, hashes):
        return np.array([h.hash.flatten().astype(int) for h in hashes])

    # Step 3: Cluster Images using DBSCAN with Hamming Distance
    def cluster_images(self, vectors, eps=0.2, min_samples=1):
        # Use DBSCAN clustering with a smaller eps value
        clustering = DBSCAN(metric='hamming', eps=eps, min_samples=min_samples)
        clusters = clustering.fit_predict(vectors)
        return clusters

    # Step 4: Clear Existing Clusters
    def clear_destination_folder(self):
        if os.path.exists(self.dest_folder):
            shutil.rmtree(self.dest_folder)
        os.makedirs(self.dest_folder, exist_ok=True)

    # Step 5: Organize Images by Cluster and return a dictionary with cluster details
    def organize_images_by_cluster(self, filenames, clusters):
        cluster_dict = {}
        unique_clusters = set(clusters)

        for cluster in unique_clusters:
            cluster_folder = os.path.join(self.dest_folder, f"cluster_{cluster}")
            os.makedirs(cluster_folder, exist_ok=True)

            cluster_count = 0
            for idx, label in enumerate(clusters):
                if label == cluster:
                    src_path = os.path.join(self.src_folder, filenames[idx])
                    dst_path = os.path.join(cluster_folder, filenames[idx])
                    shutil.copy2(src_path, dst_path)  # Use copy2 to preserve original images
                    cluster_count += 1

            # Store the cluster details in the dictionary
            cluster_dict[f"cluster_{cluster}"] = cluster_count

        return cluster_dict

    # Step 6: Perform the full comparison and clustering process
    def perform_comparison(self):
        # Clear existing clusters in the destination folder
        self.clear_destination_folder()

        # Step 1: Load and hash images
        hashes, filenames = self.load_and_hash_images()
        
        if not hashes:
            print("No images found in the source folder.")
            return
        
        # Step 2: Convert perceptual hashes to binary feature vectors
        vectors = self.hash_to_vector(hashes)
        
        # Step 3: Cluster images based on hash similarity
        clusters = self.cluster_images(vectors, eps=0.25, min_samples=1)
        
        # Step 4: Check if clustering was successful
        if len(set(clusters)) == 1:
            print("All images seem similar; consider adjusting the eps parameter.")
        else:
            print(f"Found {len(set(clusters))} clusters.")
        
        # Step 5: Organize images and get cluster summary
        cluster_summary = self.organize_images_by_cluster(filenames, clusters)
        print("Images have been organized into separate folders based on similarity.")
        print(f"Cluster Summary: {cluster_summary}")
        return cluster_summary

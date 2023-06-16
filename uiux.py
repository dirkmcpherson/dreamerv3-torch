import cv2
import torch
from tkinter import Tk, Canvas
from PIL import Image, ImageTk
import numpy as np
import random, math
from collections import defaultdict
from IPython import embed as ipshell

class UIUX:
    def __init__(self, config, world_model, goal_dec, skill_prior):
        self.config = config
        self.world_model = world_model
        self.goal_dec = goal_dec
        self.skill_prior = skill_prior

    def interface(self, feat):
        _, h_deter = feat[:self.config.dyn_stoch], feat[self.config.dyn_stoch:]
        sampled_zs = []
        sampled_imgs = []
        n_samples = 10
        for i in range(n_samples):
            skill = self.skill_prior.sample()
            stoch = skill.reshape(1, -1)
            inp = torch.concat((stoch, h_deter), dim=-1)
            img = self.world_model.heads["decoder"](inp)["image"].mode()
            # one-to-one mapping between z and image
            sampled_imgs.append(img)
            sampled_zs.append(skill)

        # TODO: Semantic Cluster. Clusters have to know their z's so we can map from selected cluster to selected z
        # make random clusters of the images
        n_clusters = 5
        clusters = [[] for _ in range(n_clusters)]
        img_cluster_map = defaultdict(list)
        for i in range(n_samples):
            cluster_idx = random.randint(0, n_clusters-1)
            img_cluster_map[cluster_idx].append(i)
            clusters[cluster_idx].append(sampled_imgs[i])

        selected_cluster = self.display_clusters(clusters)
        # map from selected cluster to selected z
        # for now select random from cluster
        z_idx = random.choice(img_cluster_map[selected_cluster])
        skill = sampled_zs[z_idx]
        return skill

    def display_clusters(self, clusters):
        root = Tk()
        radius = 200 # (0, 1)
        circle_pattern = CirclePattern(root, radius, clusters)
        root.mainloop()

        print(f"User selected {circle_pattern.selected_image_cluster}")
        return circle_pattern.selected_image_cluster

    def show_data(self, data: dict):
        assert "image" in data

        images = data["image"]

        for batch_idx, batch in enumerate(images):
            for step in batch:
                cv2.imshow(f"image batch {batch_idx}", step)
                cv2.waitKey(100)

class CirclePattern:
    def __init__(self, root, radius, clusters):
        self.root = root
        self.radius = radius
        self.clusters = clusters
        self.canvas = Canvas(self.root, width=self.root.winfo_screenwidth() // 2, height=self.root.winfo_screenheight() // 2, bg='white')
        self.canvas.pack()
        self.image_positions = []
        self.cluster_positions = []
        self.images = []

        self.selected_image_cluster = None

        # self.root.after_idle(self.load_images)
        self.root.after(10, self.load_images)

    def load_images(self):
        self.root.update()
        center = (self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2)  # Center of the canvas
        print(f"Center of canvas: {center}")
        for cluster_idx, cluster in enumerate(self.clusters):
            for i, tensor in enumerate(cluster):
                angle = 2 * math.pi * cluster_idx / len(self.clusters)
                x = center[0] + self.radius * math.cos(angle)
                y = center[1] + self.radius * math.sin(angle)
                image = Image.fromarray((tensor.numpy() * 255).astype(np.uint8))  # Convert tensor to PIL Image
                image = image.resize((100, 100), Image.ANTIALIAS)  # Resize image
                img = ImageTk.PhotoImage(image)
                self.images.append(img)
                print(f"Image {cluster_idx} loaded at position {x:1.2f} {y:1.2f}")
                img_obj = self.canvas.create_image(x, y, image=img)
            self.cluster_positions.append((x - img.width() // 2, y - img.height() // 2, x + img.width() // 2, y + img.height() // 2))
        # Bind mouse click event to canvas
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        # quit on q
        self.root.bind("q", lambda e: self.root.destroy())

    def on_canvas_click(self, event):
        for i, pos in enumerate(self.cluster_positions):
            if pos[0] <= event.x <= pos[2] and pos[1] <= event.y <= pos[3]:
                print(f"Image {i} clicked. Position in circle: {i + 1}/{len(self.clusters)}")
                self.selected_image_cluster = i
                # highlight the selected image
                self.canvas.create_rectangle(pos[0], pos[1], pos[2], pos[3], outline="red", width=5)
                self.root.after(500, self.root.destroy)

if __name__ == "__main__":
    
    # agent = 
    uiux = UIUX(None, None, None, None)
    n_clusters = 8
    clusters = [[torch.rand(64, 64, 3) for _ in range(2)] for _ in range(n_clusters)]
    uiux.display_clusters(clusters)
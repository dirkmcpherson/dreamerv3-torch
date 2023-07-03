import cv2
import torch
import torch.nn.functional as F
from tkinter import Tk, Canvas
from PIL import Image, ImageTk
import numpy as np
import random, math
import cv2
from collections import defaultdict

from sklearn.cluster import KMeans
from IPython import embed as ipshell

# unnormalize = lambda x: (x + 0.5) * 255
unnormalize = lambda x: ((x + 1) / 2) * 255 # [-1, 1] to [0, 255]

class UIUX:
    def __init__(self, config, world_model, goal_enc, goal_dec, skill_prior):
        self.config = config
        self.world_model = world_model
        self.goal_enc = goal_enc
        self.goal_dec = goal_dec
        self.skill_prior = skill_prior
        self.root = Tk()
        self.root.title("UIUX")
        self.screenw, self.screenh = self.root.winfo_screenwidth() // 2, self.root.winfo_screenheight() // 2
        # self.root.geometry("800x800")
        self.canvas = Canvas(self.root, width=self.screenw, height=self.screenh)
        self.canvas.pack()

        self.fps = config.human_fps
        self.cluster_fps = config.human_fps

        self.left_display_label = None
        self.left_image = None

        self.rectangle_label = None
        self.trust_robot_button = None

        self.selected_video_cluster = None
        self.update_clusters = True

        self.action_map = {0: "noop", 1: "forward", 2: "left", 3: "right", 4: "forward_left", 5: "forward_right"}#TODO: this should be passed in as an argument

    def imag_goals(self, latent_start, sampled_goals, actor_policy):
        '''
        Heavily based on Hierarchybehavior::_imagine_carry but with repeated starts for different goals
        '''
        dynamics = self.world_model.dynamics
        n_samples = sampled_goals.shape[0]

        # no gradients here yet
        h = latent_start["deter"].detach()
        z_stoch = latent_start["stoch"].detach()
        h = h.repeat(n_samples, 1)
        
        z = z_stoch.reshape([*z_stoch.shape[:-2], -1])
        z = z.repeat(n_samples, 1)

        inp = torch.concat([z, h, sampled_goals], dim=-1)
        # action = actor_policy(inp).sample()

        state = {"stoch": z_stoch.repeat(n_samples, 1, 1), "deter": h, "logits": None}
        
        latent_starts = [state]
        latent_actions = []

        for step in range(self.config.imag_horizon): # NOTE: static_scan is probably faster (used elsewhere)
            action = actor_policy(inp).sample()
            state = dynamics.img_step(state, action, sample=self.config.imag_sample)

            # decode the image represented by this state

            z_stoch = state["stoch"]
            z_stoch = z_stoch.reshape([*z_stoch.shape[:-2], -1])
            inp = torch.concat([z_stoch, state["deter"], sampled_goals], dim=-1)
            
            latent_starts.append(state)
            latent_actions.append(action)

        return latent_starts, latent_actions

    def trust_robot(self, event):
        self.update_clusters = False
        self.selected_video_cluster = -1

    def update_obs(self, img, obs_string):
        scale_size = 256
        posx, posy = 50, 200

        # display left and right images
        left_image = Image.fromarray((img).clip(0,255).astype(np.uint8))  # Convert tensor to PIL Image
        left_image = left_image.resize((scale_size, scale_size), Image.LANCZOS)  # Resize image
        left_img = ImageTk.PhotoImage(left_image)

        if self.left_display_label is None:
            self.left_display_label = self.canvas.create_image(posx, posy, anchor="nw")
            ## add text above the image
            self.left_text_label = self.canvas.create_text(posx, posy - 50, text=obs_string, anchor="nw", fill="black", font=("Purisa", 24))

        self.left_image = left_img
        self.canvas.itemconfig(self.left_display_label, image=left_img)
        self.canvas.itemconfig(self.left_text_label, text=obs_string)

        # add a button
        if self.trust_robot_button is None:
            self.trust_robot_button = self.canvas.create_rectangle(posx, posy + scale_size + 50, posx + 200, posy + scale_size + 100, fill="green")
            self.trust_robot_text = self.canvas.create_text(posx + 100, posy + scale_size + 75, text="Trust Robot", anchor="center", fill="black", font=("Purisa", 24))
            self.canvas.tag_bind(self.trust_robot_button, "<Button-1>", self.trust_robot)
            self.canvas.tag_bind(self.trust_robot_text, "<Button-1>", self.trust_robot)

        self.root.update()
        self.root.after(1000//self.fps)

    def interface(self, latent, actor_policy):
        n_samples = self.config.human_samples
        n_clusters = self.config.human_clusters

        z_stoch, h_deter = latent["stoch"].detach(), latent["deter"].detach()

        # NOTE: what distribution should we sample from? 
        if use_goal_enc_sampler := True:
            skill = self.goal_enc(h_deter)
            skill = skill.sample((n_samples,))
        else:
            skill = self.skill_prior.sample((n_samples,))

        skill = skill.reshape(n_samples, -1)
        dec = self.goal_dec(skill)
        # print("interface dec skill logits", dec.probs)
        goal_samples = dec.mode()
        # h_deter = h_deter.repeat(n_samples, 1)

        # NOTE: which z should we use?
        # z_stoch = z_stoch.repeat(n_samples, 1)
        # z = z_stoch
        z_from_goal = self.world_model.dynamics.get_stoch(goal_samples)
        z_from_goal = z_from_goal.reshape(n_samples, -1)
        z = z_from_goal


        '''
        For each goal decoded skill->goal we want to run an imagined worker trajectory
        '''
        imag_latents, imag_actions = self.imag_goals(latent, goal_samples, actor_policy)
        
        scaled_size = 128
        imag_images = torch.empty((0, n_samples, scaled_size, scaled_size, 1 if self.config.grayscale else 3))  

        for imag_latent in imag_latents: # NOTE: there's a way to do this without a loop taht's probably faster
            z_stoch, h_deter = imag_latent["stoch"], imag_latent["deter"]
            z = z_stoch.reshape([*z_stoch.shape[:-2], -1])
            inp = torch.concat([z, h_deter], dim=-1)
            img = self.world_model.heads["decoder"](inp.unsqueeze(0))["image"].mode().detach().cpu()
            # scale up the image
            img = F.interpolate(img, size=(scaled_size, scaled_size, 1 if self.config.grayscale else 3), mode="nearest")

            imag_images = torch.cat([imag_images, img], dim=0)


        # Make videos of the imagined trajectories
        # ipshell()

        ## Make a tkinter gui that will play the videos

        # for video_idx in range(imag_images.shape[1]): # each video is for a different goal
        #     root = Tk()
        #     root.title(f"{video_idx}")
        #     root.geometry("128x128")
        #     canvas = Canvas(root, width=128, height=128)
        #     canvas.pack()
        #     photo_imgs = []
        #     for frame_idx in range(imag_images.shape[0]):
        #         print(frame_idx, video_idx)
        #         img = imag_images[frame_idx, video_idx].numpy().squeeze()
        #         img = unnormalize(img)
        #         img = Image.fromarray(img.astype(np.uint8))
        #         photo_img = ImageTk.PhotoImage(img)
        #         canvas.create_image(0, 0, image=photo_img, anchor="nw")
        #         photo_imgs.append(photo_img)
        #         root.update()
        #         root.after(10)

        #     # canvas.delete("all")
        #     for photo_img in photo_imgs:
        #         photo_img.__del__()

        #     canvas.destroy()
        #     root.destroy()



        inp = torch.concat((z, goal_samples), dim=-1)
        img = self.world_model.heads["decoder"](inp.unsqueeze(0))["image"].mode().squeeze(0).detach().cpu()
    
        # outputs from -1 to 1, so we need to scale it back otherwise we have to truncate
        # NOTE: do we scale based on expected output range? or based on this batch?
        img_scaled = ((img - img.min()) / (img.max() - img.min())) * 255
        # img = (img + 1) / (2) # [-1,1] to [0,1]

        img = img_scaled
        
        feat = self.world_model.dynamics.get_feat(latent)
        inp = feat.unsqueeze(0)
        decoded_img = self.world_model.heads["decoder"](inp)["image"].mode()
        decoded_img = decoded_img.detach().cpu().numpy().squeeze().squeeze()
        decoded_img = unnormalize(decoded_img)
        # show = np.concatenate([obs_image, decoded_img, skill_to_img[0].squeeze()], axis=0)
        # cv2.imshow("show", show)
        # cv2.waitKey(10)

        # TODO: Semantic Cluster. Clusters have to know their z's so we can map from selected cluster to selected z
        # make random clusters of the images
        
        # get the mean rgb values for each image
        mean_rgb = img.mean(dim=(1, 2))
        std_rgb = img.std(dim=(1, 2))

        # print(f"mean mean rgb: {mean_rgb.mean():1.3f} std mean rgb: {mean_rgb.std():1.3f}")

        # cluster the images based on the mean rgb values
        actions = np.array([a.detach().cpu().numpy() for a in imag_actions]) # (n_frames, n_samples, action_dim)
        actions = np.array([a.argmax(axis=-1) for a in actions]) # (n_frames, n_samples) # NOTE: convert to argmax
        actions = actions.T # (n_samples, n_frames) for clustering
        # ipshell()
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(actions)
        labels = kmeans.labels_
        # labels = [i%n_clusters for i in range(len(img))]

        clusters = [[] for _ in range(n_clusters)]
        clustered_actions = [[] for _ in range(n_clusters)] # track the actions for each cluster
        traj_cluster_map = defaultdict(list)
        for traj_idx, label in enumerate(labels):
            # print(f"traj_idx: {traj_idx} label: {label}")
            traj_cluster_map[label].append(traj_idx)
            clusters[label].append(imag_images[:, traj_idx].detach().cpu().numpy().squeeze())
            
            traj_actions = np.array([act[traj_idx].detach().cpu().numpy().squeeze() for act in imag_actions]) # pull the frames from trajectory traj_idx
            clustered_actions[label].append(traj_actions)

        for cluster_idx in range(n_clusters):
            print(f"cluster_idx: {cluster_idx} n_trajs: {len(traj_cluster_map[cluster_idx])}")
            # assert clustered_actions[cluster_idx][0].shape[0] == len(traj_cluster_map[cluster_idx])
            
        n_frames = imag_images.shape[0]
        video_list = []
        video_texts = []
        for cluster_idx in range(n_clusters):
            frame_list = []
            image_texts = []
            random_cluster_example = 0 # NOTE: how should we select from the cluster?
            if len(clusters[cluster_idx]) == 0:
                print(f"WARN: cluster_idx: {cluster_idx} is empty")
                continue

            cluster_example = clusters[cluster_idx][random_cluster_example]
            cluster_actions_example = clustered_actions[cluster_idx][random_cluster_example]
            # print(f"cluster example shape: {cluster_example.shape}, cluster_actions_example shape: {cluster_actions_example.shape}")
            for frame_idx in range(n_frames):
                # print(f"cluster_idx: {cluster_idx} frame_idx: {frame_idx}")
                try:
                    frame_list.append(cluster_example[frame_idx])
                except:
                    ipshell()

                if frame_idx < len(cluster_actions_example): # no action on the last frame
                    action_idx = cluster_actions_example[frame_idx].argmax()
                    action_str = self.action_map[action_idx]
                else:
                    action_str = "n/a"

                image_texts.append(f"{action_str}")

            video_texts.append(image_texts)
            video_list.append(frame_list)


        # ipshell()

        ## Sample videos:
        # n_frames = imag_images.shape[0]
        # n_videos = 20
        # video_list = []
        # video_texts = []
        # for video_idx in range(n_videos):
        #     frame_list = []
        #     image_texts = []
        #     for frame_idx in range(n_frames):
        #         frame_list.append(imag_images[frame_idx, video_idx].numpy().squeeze())

        #         if frame_idx < len(imag_actions): # no action on the last frame
        #             action_idx = imag_actions[frame_idx][video_idx].detach().cpu().numpy().squeeze().argmax()
        #             action_str = self.action_map[action_idx]
        #         else:
        #             action_str = "n/a"
        #         image_texts.append(f"{frame_idx} {action_str}")

        #     video_texts.append(image_texts)
        #     video_list.append(frame_list)
    
        selected_cluster = self.display_clusters(video_list, video_texts)

        if selected_cluster < 0:
            return None, "Trust Robot"
        else:
            # z_idx = selected_cluster # NOTE: random.choice(traj_cluster_map[selected_cluster])
            z_idx = random.choice(traj_cluster_map[selected_cluster])
            skill = skill[z_idx]
            return skill, None

    def display_clusters(self, video_clusters, video_texts):
        self.update_clusters = True
        # radius = 200 # (0, 1)
        # circle_pattern = CirclePattern(root, radius, clusters)

        # Display the videos in a grid on the canvas
        # calculate the size of the grid
        n_videos = len(video_clusters)
        n_rows = int(np.sqrt(n_videos))
        n_cols = int(np.ceil(n_videos / n_rows))

        # calculate the size of each video
        video_w, video_h = video_clusters[0][0].shape[0:2]

        # calculate the size of the grid in pixels
        grid_w = n_cols * video_w
        grid_h = n_rows * video_h

        # calculate the position of the grid in pixels
        # grid_x = (screenw) // 2
        # grid_y = (screenh) // 2
        grid_x = (self.screenw - grid_w) // 2
        grid_y = (self.screenh - grid_h) // 3

        grid_spacing = 50

        video_positions = []
        video_bounds = []
        for i in range(len(video_clusters)):
            x = grid_x + (i % n_cols) * video_w + (i % n_cols) * grid_spacing
            y = grid_y + (i // n_cols) * video_h + (i // n_cols) * grid_spacing
            video_positions.append((x, y))
            video_bounds.append((x, y, x + video_w, y + video_h))

        def on_canvas_click(event):
            print(f"Clicked at {event.x}, {event.y}")
            for i, pos in enumerate(video_bounds):
                if pos[0] <= event.x <= pos[2] and pos[1] <= event.y <= pos[3]:
                    print(f"Image {i} clicked. Position {video_positions[i]}")
                    self.selected_video_cluster = i
                    # highlight the selected image
                    self.rectangle_label = self.canvas.create_rectangle(pos[0], pos[1], pos[2], pos[3], outline="red", width=5)
                    # self.root.after(500, self.root.destroy)
                    self.update_clusters = False

        def display_videos_on_canvas(root, canvas, videos, video_positions, video_texts):
            """
            Displays a list of numpy images as a looping video on a tkinter canvas.

            Args:
                root (Tk): The tkinter root object.
                canvas (Canvas): The tkinter canvas object.
                videos (List[List[np.ndarray]]): A list of lists of numpy arrays representing the frames of each video.
                fps (int): The frames per second to display the video at. Defaults to 30.
            """
            assert len(videos) == len(video_positions), "The number of videos must match the number of video positions."
            # create a list to hold the PhotoImage objects for each video
            photo_images_list = []
            photo_texts_list = []

            # create a tkinter label and list of PhotoImage objects for each video
            video_labels = []
            text_labels = []
            for idx, video in enumerate(videos):
                photo_images = []
                photo_texts = []
                # video_label = canvas.create_image(0, idx*image_h, anchor="nw")
                x,y = video_positions[idx]
                video_label = canvas.create_image(x, y, anchor="nw")
                video_labels.append(video_label)
                text_label = canvas.create_text(x - grid_spacing // 4, y - grid_spacing // 2, anchor="nw", text=video_texts[idx][0], font=("Arial", 16, "bold"), fill="black")
                text_labels.append(text_label)
                for img_idx, img in enumerate(video):
                    img = (img + 0.5)*255
                    img = img.clip(0, 255) # NOTE: this is one way to do it, otherwise we could scale the image
                    pil_img = Image.fromarray(img.astype(np.uint8))
                    photo_image = ImageTk.PhotoImage(pil_img)
                    photo_images.append(photo_image)
                    photo_texts.append(video_texts[idx][img_idx])
                photo_images_list.append(photo_images)
                photo_texts_list.append(photo_texts)

            # display the videos on the canvas
            while self.update_clusters:
                for i, photo_images in enumerate(photo_images_list):
                    canvas.itemconfig(video_labels[i], image=photo_images[0])
                    canvas.itemconfig(text_labels[i], text=photo_texts_list[i][0])
                    # canvas.create_text(video_positions[i][0], video_positions[i][1] + video_h, text=photo_texts_list[i][0], anchor="nw", fill="black", font=("Purisa", 24))

                    photo_images.append(photo_images.pop(0))
                    photo_texts_list[i].append(photo_texts_list[i].pop(0))

                canvas.itemconfig(self.left_display_label, image=self.left_image)
                root.update()
                root.after(int(1000/self.cluster_fps))

            # delete the text labels
            for text_label in text_labels:
                canvas.delete(text_label)

        # display left and right images
        # left_image = Image.fromarray((left_side_image).clip(0,255).astype(np.uint8))  # Convert tensor to PIL Image
        # left_image = left_image.resize((256, 256), Image.LANCZOS)  # Resize image
        # left_img = ImageTk.PhotoImage(left_image)
        # left_img_obj = self.canvas.create_image(0, 300, image=left_img, anchor="nw")

        # right_image = Image.fromarray((right_side_image).clip(0,255).astype(np.uint8))  # Convert tensor to PIL Image
        # right_image = right_image.resize((256, 256), Image.LANCZOS)  # Resize image
        # right_img = ImageTk.PhotoImage(right_image)
        # right_img_obj = self.canvas.create_image(1400, 300, image=right_img)

        # display videos
        self.canvas.bind("<Button-1>", on_canvas_click)
        self.root.bind("q", lambda e: self.root.destroy())

        display_videos_on_canvas(self.root, self.canvas, video_clusters, video_positions, video_texts)
        self.canvas.delete(self.rectangle_label)

        # self.root.mainloop()

        print(f"User selected {self.selected_video_cluster}")
        return self.selected_video_cluster

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
        self.cluster_bounds = []
        self.images = []

        self.selected_image_cluster = None

        # self.root.after_idle(self.load_images)
        self.root.after(10, self.load_images)

    def spatial_calculations(self):
        self.root.update()
        center = (self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2)  # Center of the canvas
        print(f"Center of canvas: {center}")
        clusters = []
        for cluster_idx, cluster in enumerate(self.clusters):
            image = None
            for _, tensor in enumerate(cluster):
                angle = 2 * math.pi * cluster_idx / len(self.clusters)
                x = center[0] + self.radius * math.cos(angle)
                y = center[1] + self.radius * math.sin(angle)

            clusters.append(cluster_idx)
            self.cluster_bounds.append((x - self.img_width // 2, y - self.img_height // 2, x + self.img_width // 2, y + self.img_height // 2))
        
        # Bind mouse click event to canvas
        # quit on q
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.root.bind("q", lambda e: self.root.destroy())

        # self.selected_image_cluster = valid_clusters[0] # NOTE: temporary random selection
        # self.root.after(500, self.root.destroy())

    def load_images(self):
        self.root.update()
        center = (self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2)  # Center of the canvas
        print(f"Center of canvas: {center}")
        valid_clusters = []
        for cluster_idx, cluster in enumerate(self.clusters):
            image = None
            for _, tensor in enumerate(cluster):
                angle = 2 * math.pi * cluster_idx / len(self.clusters)
                x = center[0] + self.radius * math.cos(angle)
                y = center[1] + self.radius * math.sin(angle)
                img_array = tensor.numpy().astype(np.uint8)
                image = Image.fromarray(img_array)  # Convert tensor to PIL Image
                image = image.resize((128, 128), Image.ANTIALIAS)  # Resize image
                img = ImageTk.PhotoImage(image)
                self.images.append(img)
                img_obj = self.canvas.create_image(x, y, image=img)

                # Add text about cluster and position
                self.canvas.create_text(x, y-100, text=f"Cluster {cluster_idx}", font=("Arial", 16  ), fill="black")
            if image is None:
                print(f"Cluster {cluster_idx} is empty")
            else:
                valid_clusters.append(cluster_idx)
                self.cluster_bounds.append((x - img.width() // 2, y - img.height() // 2, x + img.width() // 2, y + img.height() // 2))
        # Bind mouse click event to canvas
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        # quit on q
        self.root.bind("q", lambda e: self.root.destroy())

        # self.selected_image_cluster = valid_clusters[0] # NOTE: temporary random selection
        # self.root.after(500, self.root.destroy())

    def on_canvas_click(self, event):
        for i, pos in enumerate(self.cluster_bounds):
            if pos[0] <= event.x <= pos[2] and pos[1] <= event.y <= pos[3]:
                print(f"Image {i} clicked. Position in circle: {i + 1}/{len(self.clusters)}")
                self.selected_image_cluster = i
                # highlight the selected image
                self.canvas.create_rectangle(pos[0], pos[1], pos[2], pos[3], outline="red", width=5)
                # self.root.after(500, self.root.destroy)
                self.update_clusters = False

if __name__ == "__main__":
    
    # agent = 
    uiux = UIUX(None, None, None, None)
    n_clusters = 8
    clusters = [[torch.rand(64, 64, 3) for _ in range(2)] for _ in range(n_clusters)]
    uiux.display_clusters(clusters)
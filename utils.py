# import torch
# import numpy as np
# import cv2
# import torch.nn.functional as F
# from lime import lime_image

# class GradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
        
#         def forward_hook(module, input, output):
#             self.activations = output.detach()
        
#         def backward_hook(module, grad_in, grad_out):
#             self.gradients = grad_out[0].detach()
        
#         target_layer.register_forward_hook(forward_hook)
#         target_layer.register_backward_hook(backward_hook)
    
#     def generate_cam(self, input_image, target_class=None):
#         output = self.model(input_image)
        
#         if target_class is None:
#             target_class = output.argmax(dim=1).item()
        
#         self.model.zero_grad()
#         output[:, target_class].backward()
        
#         weights = torch.mean(self.gradients, dim=(2, 3))
#         cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32)
        
#         for i, w in enumerate(weights[0]):
#             cam += w * self.activations[0, i]
        
#         cam = torch.relu(cam)
#         cam = cam.detach().cpu().numpy()
#         cam = cv2.resize(cam, input_image.shape[2:][::-1])
#         cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-7)
        
#         return cam, target_class

# class LimeExplainer:
#     def __init__(self, model, device='cuda'):
#         self.model = model
#         self.device = device
    
#     def batch_predict(self, images):
#         self.model.eval()
#         if images.dtype == np.uint8:
#             images = images.astype(np.float32) / 255.0
#         batch = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(self.device)
#         with torch.no_grad():
#             output = self.model(batch)
#             probs = F.softmax(output, dim=1)
#         return probs.cpu().numpy()

# def generate_gradcam_visualization(image, model, target_layer):
#     gradcam = GradCAM(model, target_layer)
#     cam, _ = gradcam.generate_cam(image)
#     return cam

# def generate_lime_explanation(image, model, device='cuda'):
#     explainer = LimeExplainer(model, device)
#     explanation = lime_image.LimeImageExplainer().explain_instance(
#         np.array(image),
#         explainer.batch_predict,
#         top_labels=3,
#         hide_color=0,
#         num_samples=100
#     )
#     return explanation

import torch
import numpy as np
import cv2
import torch.nn.functional as F
from lime import lime_image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import pickle
from PIL import Image

# class GradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
        
#         # Register hooks
#         def forward_hook(module, input, output):
#             self.activations = output.detach()
            
#         def backward_hook(module, grad_in, grad_out):
#             self.gradients = grad_out[0].clone()  # Clone to avoid in-place modifications
        
#         # Ensure target_layer is not a tuple
#         if isinstance(target_layer, tuple):
#             target_layer = target_layer[0]
            
#         # Clear existing hooks
#         target_layer._forward_hooks.clear()
#         target_layer._backward_hooks.clear()
        
#         # Register new hooks
#         self.forward_hook = target_layer.register_forward_hook(forward_hook)
#         self.backward_hook = target_layer.register_backward_hook(backward_hook)
    
#     def __del__(self):
#         # Clean up hooks when the object is deleted
#         try:
#             self.forward_hook.remove()
#             self.backward_hook.remove()
#         except:
#             pass
    
#     def generate_cam(self, input_image, target_class=None):
#         """Generate GradCAM for input image"""
#         # Ensure model is in eval mode
#         self.model.eval()
        
#         # Get model prediction
#         with torch.enable_grad():
#             output = self.model(input_image)
            
#             if target_class is None:
#                 target_class = output.argmax(dim=1).item()
            
#             # Zero gradients
#             self.model.zero_grad()
            
#             # Create a one-hot target tensor
#             one_hot = torch.zeros_like(output)
#             one_hot[0, target_class] = 1
            
#             # Backward pass for target class
#             output.backward(gradient=one_hot, retain_graph=True)
            
#             # Get weights
#             weights = torch.mean(self.gradients, dim=(2, 3))
            
#             # Move weights to same device as activations
#             weights = weights.to(self.activations.device)
            
#             # Generate cam
#             cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32, device=self.activations.device)
#             for i, w in enumerate(weights[0]):
#                 cam += w * self.activations[0, i]
            
#             # Apply ReLU and normalize
#             cam = torch.relu(cam)
#             cam = cam.detach().cpu().numpy()
            
#             # Resize and normalize
#             cam = cv2.resize(cam, input_image.shape[2:][::-1])
#             cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-7)
            
#             return cam, target_class, output.detach()

# def preprocess_image_(image_path, target_size=(224, 224)):
#     """Preprocess image for model input"""
#     # Load and resize image
#     image = Image.open(image_path).convert('RGB')
#     image = image.resize(target_size)
    
#     # Convert to numpy and normalize
#     image_np = np.array(image) / 255.0
    
#     # Convert to tensor and add batch dimension
#     image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
#     image_tensor = image_tensor.unsqueeze(0)
    
#     return image_tensor, image_np

# def generate_gradcam_visualization(model, image_path, target_layer, class_names, device='cuda'):
#     """Generate GradCAM visualization for a model"""
#     try:
#         # Preprocess image
#         image_tensor, image_np = preprocess_image_(image_path)
#         image_tensor = image_tensor.to(device)
#         model = model.to(device)
        
#         # Initialize GradCAM
#         grad_cam = GradCAM(model, target_layer)
        
#         # Generate CAM
#         cam, pred_class, output = grad_cam.generate_cam(image_tensor)
        
#         # Get top predictions
#         with torch.no_grad():
#             probs = torch.nn.functional.softmax(output, dim=1)
#             top_probs, top_classes = torch.topk(probs[0], k=3)
        
#         # Create heatmap
#         heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
#         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#         heatmap = np.float32(heatmap) / 255
        
#         # Create superimposed image
#         superimposed = 0.6 * image_np + 0.4 * heatmap
#         superimposed = superimposed / (superimposed.max() + 1e-7)
        
#         return superimposed, pred_class, {
#             'cam': cam,
#             'top_classes': top_classes.cpu().numpy(),
#             'top_probs': top_probs.cpu().numpy()
#         }
#     except Exception as e:
#         print(f"Error generating GradCAM for model: {str(e)}")
#         raise

# def plot_multiple_gradcams(image_path, models, model_names, target_layers, class_names, 
#                           save_path=None, device='cuda'):
#     """Generate and plot GradCAM for multiple models"""
#     num_models = len(models)
    
#     plt.figure(figsize=(15, 5*num_models))
    
#     # Original image for reference
#     img = Image.open(image_path).convert('RGB')
#     img_np = np.array(img.resize((224, 224))) / 255.0
    
#     plt.subplot(num_models, 3, 1)
#     plt.imshow(img_np)
#     plt.title('Original Image')
#     plt.axis('off')
    
#     # Generate GradCAM for each model
#     for idx, (model, name, target_layer) in enumerate(zip(models, model_names, target_layers)):
#         print(f"\nGenerating GradCAM for {name}...")
        
#         try:
#             # Ensure the model is in eval mode
#             model.eval()
            
#             superimposed, pred_class, details = generate_gradcam_visualization(
#                 model,
#                 image_path,
#                 target_layer,
#                 class_names,
#                 device
#             )
            
#             # Plot heatmap
#             plt.subplot(num_models, 3, idx*3 + 2)
#             plt.imshow(details['cam'], cmap='jet')
#             plt.title(f'{name} - Activation Map')
#             plt.axis('off')
            
#             # Plot superimposed
#             plt.subplot(num_models, 3, idx*3 + 3)
#             plt.imshow(superimposed)
#             plt.title(f'{name} - Superimposed\nPredicted: {class_names[pred_class]}')
#             plt.axis('off')
            
#             # Print top predictions
#             print(f"\nTop predictions for {name}:")
#             for cls, prob in zip(details['top_classes'], details['top_probs']):
#                 print(f"{class_names[cls]}: {prob:.3f}")
        
#         except Exception as e:
#             print(f"Error processing model {name}: {str(e)}")
#             continue
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight', dpi=300)
    

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Initializes GradCAM with the specified model and target layer.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].clone()  # Clone to avoid in-place modifications

        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(forward_hook)
        self.backward_hook = target_layer.register_backward_hook(backward_hook)

    def __del__(self):
        # Clean up hooks when the object is deleted
        try:
            self.forward_hook.remove()
            self.backward_hook.remove()
        except:
            pass

    def generate_cam(self, input_tensor, target_class=None):
        """
        Generates the Class Activation Map (CAM) for the input tensor.
        """
        self.model.eval()

        with torch.enable_grad():
            output = self.model(input_tensor)
            
            if target_class is None:
                target_class = output.argmax(dim=1).item()

            self.model.zero_grad()
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1
            output.backward(gradient=one_hot, retain_graph=True)

            weights = torch.mean(self.gradients, dim=(2, 3))  # Global Average Pooling
            cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32)

            for i, w in enumerate(weights[0]):
                cam += w * self.activations[0, i]

            cam = torch.relu(cam)
            cam = cam.detach().cpu().numpy()
            cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-7)

            return cam, target_class
        
# utils.py

def plot_multiple_gradcams(image, models, model_names, target_layers, class_names, device='cpu'):
    """
    Generate and plot GradCAM visualizations for multiple models on a given image.

    Args:
        image (PIL.Image): The input image.
        models (list): List of models to generate GradCAM for.
        model_names (list): Corresponding names for the models.
        target_layers (list): Target layers for each model.
        class_names (list): List of class names.
        device (str): Device to run the computations on ('cpu').

    Returns:
        matplotlib.figure.Figure: The generated plot.
    """
    num_models = len(models)
    
    # Ensure the image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Preprocess the image
    image = image.resize((224, 224))
    image_np = np.array(image) / 255.0
    
    # Debug: Check image shape
    print(f"Image shape after conversion and resizing: {image_np.shape}")  # Should be (224, 224, 3)
    
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    fig, axes = plt.subplots(num_models, 2, figsize=(12, 6 * num_models))
    
    # If only one model, make axes iterable
    if num_models == 1:
        axes = [axes]
    
    for idx, (model, name, target_layer) in enumerate(zip(models, model_names, target_layers)):
        grad_cam = GradCAM(model, target_layer)
        cam, pred_class = grad_cam.generate_cam(image_tensor)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255
        
        # Superimpose heatmap on original image
        superimposed = 0.6 * image_np + 0.4 * heatmap
        superimposed = superimposed / (superimposed.max() + 1e-7)
        
        # Plot CAM
        axes[idx][0].imshow(cam, cmap='jet')
        axes[idx][0].set_title(f'{name} - Activation Map')
        axes[idx][0].axis('off')
        
        # Plot superimposed image
        axes[idx][1].imshow(superimposed)
        axes[idx][1].set_title(f'{name} - Superimposed\nPredicted: {class_names[pred_class]}')
        axes[idx][1].axis('off')
    
    plt.tight_layout()
    return fig




class LimeModelExplainer:
    def __init__(self, model, class_names, device='cpu'):
        self.model = model
        self.device = device
        self.class_names = class_names
        print(f"LIME Explainer initialized")
    
    def batch_predict(self, images):
        """Batch prediction for LIME"""
        self.model.eval()
        
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / 255.0
            
        images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
        return probs.cpu().numpy()
    
    def explain_image(self, image, num_samples=1000, top_labels=4, hide_color=0):
        """Generate LIME explanation for a single image"""
        print("Generating LIME explanation...")
        
        explainer = lime_image.LimeImageExplainer(verbose=False)
        
        explanation = explainer.explain_instance(
            image,
            lambda x: self.batch_predict(x),
            top_labels=top_labels,
            hide_color=hide_color,
            num_samples=num_samples
        )
        
        # Get predictions for the original image
        orig_pred = self.batch_predict(np.expand_dims(image, axis=0))[0]
        
        return explanation, orig_pred

def plot_lime_results(explanation, predictions, image, class_names, num_features=5, save_path=None):
    """Plot LIME explanations with the correct probability access"""
    top_labels = explanation.top_labels
    
    plt.figure(figsize=(20, 4))
    
    # Original image
    plt.subplot(1, 5, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Original Image')
    
    # Plot explanations for top 4 classes
    for idx, label in enumerate(top_labels[:4], 1):
        temp, mask = explanation.get_image_and_mask(
            label,
            positive_only=True,
            num_features=num_features,
            hide_rest=False
        )
        
        plt.subplot(1, 5, idx + 1)
        plt.imshow(mark_boundaries(temp, mask))
        plt.axis('off')
        plt.title(f'Class: {class_names[label]}\nProb: {predictions[label]:.3f}')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
   

def explain_images_from_paths(model, image_paths, class_names, device='cuda', 
                            save_path=None, target_size=(224, 224)):
    """Generate LIME explanations for images from paths"""
    print("\n1. Starting image loading and explanation process...")
    
    explainer = LimeModelExplainer(model, class_names, device)
    
    for idx, path in enumerate(image_paths):
        try:
            print(f"\nProcessing image {idx+1}/{len(image_paths)}: {path}")
            
            # Load and preprocess image
            img = Image.open(path).convert('RGB')
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            img_np = np.array(img)
            
            print("Generating explanation...")
            explanation, predictions = explainer.explain_image(
                img_np,
                num_samples=1000,
                top_labels=4
            )
            
            # Plot results
            save_path_i = f"{save_path}_{idx}.png" if save_path else None
            plot_lime_results(explanation, predictions, img_np, class_names, save_path=save_path_i)
            
            # Print top predictions
            top_k = 3
            top_indices = np.argsort(predictions)[-top_k:][::-1]
            print("\nTop predictions:")
            for i in top_indices:
                print(f"{class_names[i]}: {predictions[i]:.3f}")
            
            print(f"Explanation complete for image {idx+1}")
            
        except Exception as e:
            print(f"Error processing image {path}: {str(e)}")
            continue
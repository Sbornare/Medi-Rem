import os
import json
import re
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from paddleocr import PaddleOCR
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import f1_score
import logging
import matplotlib.pyplot as plt
import gc
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)8s] %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('prescription_process.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class PrescriptionDataset(Dataset):
    def __init__(self, image_dir, cache_dir, transform=None, force_reload=False, 
                 batch_process=True, batch_size=50, image_size=(400, 600)):
        self.image_dir = image_dir
        self.cache_dir = cache_dir
        self.transform = transform
        self.force_reload = force_reload
        self.batch_process = batch_process
        self.batch_size = batch_size
        self.image_size = image_size
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Get list of image files
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Process images and cache results
        if force_reload or not self._check_cache_exists():
            logger.info("Processing images and caching results...")
            self._preprocess_images()
            
        # Load cached results
        self.cached_results = self._load_cached_results()
        
        # Print dataset statistics
        logger.info("\nDataset Statistics:")
        logger.info(f"Total images: {len(self.image_files)}")
        logger.info(f"Cached results: {len(self.cached_results)}")
        
        # Print sample label distribution
        labels = [result['label'] for result in self.cached_results.values()]
        labels = np.array(labels)
        logger.info("\nLabel Distribution:")
        for i in range(labels.shape[1]):
            positive_count = np.sum(labels[:, i] == 1)
            logger.info(f"Class {i}: {positive_count} positive samples ({positive_count/len(labels)*100:.2f}%)")

    def _check_cache_exists(self):
        # Check if a sample of files exists to speed up initialization
        sample_size = min(100, len(self.image_files))
        sample_files = self.image_files[:sample_size]
        
        return all(
            os.path.exists(os.path.join(self.cache_dir, f"{os.path.splitext(img)[0]}.json"))
            for img in sample_files
        )

    def _preprocess_images(self):
        successful = 0
        failed = 0
        
        # Initialize OCR only once
        ocr = PaddleOCR(
            use_angle_cls=False,
            lang='en',
            show_log=False,
            use_gpu=torch.cuda.is_available(),
            enable_mkldnn=True,
            cpu_threads=4
        )
        
        # Process images in batches
        if self.batch_process:
            remaining_files = list(self.image_files)
            
            while remaining_files:
                # Take a batch of files
                batch_files = remaining_files[:self.batch_size]
                remaining_files = remaining_files[self.batch_size:]
                
                for img_file in tqdm(batch_files, desc=f"Processing batch ({len(batch_files)} images)"):
                    try:
                        # Skip if already cached
                        cache_path = os.path.join(self.cache_dir, f"{os.path.splitext(img_file)[0]}.json")
                        if os.path.exists(cache_path):
                            successful += 1
                            continue
                            
                        # Process image
                        image_path = os.path.join(self.image_dir, img_file)
                        result = self._process_single_image(image_path, ocr)
                        
                        # Cache result
                        with open(cache_path, 'w') as f:
                            json.dump(result, f)
                        
                        successful += 1
                            
                    except Exception as e:
                        logger.error(f"Error processing {img_file}: {str(e)}")
                        failed += 1
                        continue
                
                # Force garbage collection between batches
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            # Process images one by one (original approach)
            for img_file in tqdm(self.image_files, desc="Processing images"):
                try:
                    # Skip if already cached
                    cache_path = os.path.join(self.cache_dir, f"{os.path.splitext(img_file)[0]}.json")
                    if os.path.exists(cache_path):
                        successful += 1
                        continue
                        
                    # Process image
                    image_path = os.path.join(self.image_dir, img_file)
                    result = self._process_single_image(image_path, ocr)
                    
                    # Cache result
                    with open(cache_path, 'w') as f:
                        json.dump(result, f)
                    
                    successful += 1
                        
                except Exception as e:
                    logger.error(f"Error processing {img_file}: {str(e)}")
                    failed += 1
                    continue
        
        logger.info(f"\nProcessing complete:")
        logger.info(f"Successfully processed: {successful}")
        logger.info(f"Failed to process: {failed}")
        logger.info(f"Success rate: {(successful/(successful+failed))*100:.2f}%")

    def _process_single_image(self, image_path, ocr):
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                return {'text': [], 'label': [0] * 7}
                
            # Use a smaller resize to reduce memory usage
            image = cv2.resize(image, self.image_size)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply simpler preprocessing to save memory
            # Skip adaptive histogram equalization (CLAHE) which is memory-intensive
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Use simple thresholding instead of adaptive thresholding
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Perform OCR with optimized parameters
            try:
                result = ocr.ocr(binary, cls=False)
                # Force garbage collection after OCR to free memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"OCR failed for {image_path}: {str(e)}")
                return {'text': [], 'label': [0] * 7}
            
            # Handle None or empty results
            if result is None or len(result) == 0 or not result[0]:
                logger.warning(f"No OCR results for {image_path}")
                return {'text': [], 'label': [0] * 7}
                
            # Extract text and confidence with validation
            texts = []
            confidences = []
            try:
                for line in result[0]:
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                        confidence = line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 0.0
                        texts.append(text)
                        confidences.append(confidence)
            except Exception as e:
                logger.warning(f"Error extracting text from OCR result for {image_path}: {str(e)}")
                return {'text': [], 'label': [0] * 7}
            
            # Generate labels based on OCR text
            label = self._generate_labels(texts)
            
            return {
                'text': texts,
                'confidence': confidences,
                'label': label
            }
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return {'text': [], 'label': [0] * 7}

    @lru_cache(maxsize=128)
    def _normalize_text(self, text):
        """Normalize text for better matching, with caching for performance"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove common prefixes and suffixes
        prefixes = ['tab', 'tablet','TAB','Tab','Cap','CAP', 'cap', 'capsule', 'syrup', 'SYR','Syrup','Syr','suspension', 'INJ','Inj','inj', 'injection']
        for prefix in prefixes:
            if text.startswith(prefix + ' '):
                text = text[len(prefix):].strip()
        
        # Remove special characters and extra spaces
        text = ''.join(c for c in text if c.isalnum() or c.isspace())
        text = ' '.join(text.split())
        
        return text

    def _generate_labels(self, texts):
        # Initialize label vector
        label = [0] * 7
        
        medicine_pattern = re.compile(r"([a-zA-Z\s]+(?:tab|cap|tablet|capsule))\s*(\d+mg)\s*(\d+x|\d+\s*times)\s*(\d+\s*days|\d+\s*week)")
        syrup_pattern = re.compile(r"([a-zA-Z\s]+(?:syrup))\s*(\d+x|\d+\s*times)\s*(\d+\s*days|\d+\s*week)")

        
        # Common medication form indicators with variations
        forms = {
            0: {  # Tablets
                'keywords': ['tablet','Tablet', 'Tab','TAB','tab','PILL','Pill', 'pill'],
                'variations': ['tablets', 'tabs', 'pills'],
                'regex': medicine_pattern
            },
            1: {  # Capsules
                'keywords': ['capsule','Capsule','CAP','Cap', 'cap'],
                'variations': ['capsules', 'caps'],
                'regex': medicine_pattern
            },
            2: {  # Syrups
                'keywords': ['syrup','Syrup','SYR','Syr', 'suspension'],
                'variations': ['syrups', 'suspensions'],
                'regex': syrup_pattern
            },
            3: {  # Injections
                'keywords': ['injection','Injection','INJECTION','INJ', 'inj'],
                'variations': ['injections', 'injs']
            },
            4: {  # Drops
                'keywords': ['drops', 'eye drops'],
                'variations': ['drop', 'eyedrops']
            },
            5: {  # Topical
                'keywords': ['cream', 'ointment'],
                'variations': ['creams', 'ointments']
            },
            6: {  # Inhalers/Sprays
                'keywords': ['inhaler', 'spray'],
                'variations': ['inhalers', 'sprays']
            }
        }
        
        # Process each text line
        for text in texts:
            # Normalize text
            normalized_text = self._normalize_text(text)
            
            # Check for medication forms
            for form_id, form_info in forms.items():
            # Check main keywords
                if any(keyword in normalized_text for keyword in form_info['keywords']):
                    label[form_id] = 1
                    break
                
                # Check variations
                if any(variation in normalized_text for variation in form_info['variations']):
                    label[form_id] = 1
                    break
                
                # Check for exact matches (case-insensitive)
                if any(keyword.lower() in text.lower() for keyword in form_info['keywords']):
                    label[form_id] = 1
                    break
                
                # Check for variations with exact matches
                if any(variation.lower() in text.lower() for variation in form_info['variations']):
                    label[form_id] = 1
                    break
        
        return label

    def _load_cached_results(self):
        cached_results = {}
        for img_file in self.image_files:
            cache_path = os.path.join(self.cache_dir, f"{os.path.splitext(img_file)[0]}.json")
            try:
                if os.path.exists(cache_path):
                    with open(cache_path, 'r') as f:
                        cached_results[img_file] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache for {img_file}: {str(e)}")
                continue
        return cached_results

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, img_file)
        
        # Load image with PIL and handle memory issue
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            # Return a black image in case of error
            image = torch.zeros(3, 224, 224)
        
        # Get cached result
        result = self.cached_results.get(img_file, {'label': [0] * 7})
        
        return {
            'image': image,
            'label': torch.tensor(result['label'], dtype=torch.float32)
        }

class MediRemModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # Use a smaller model initially
        self.resnet = models.resnet18(pretrained=True)
        
        # Modify the final layers
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.resnet(x)

class PrescriptionTrainer:
    def __init__(self, model, train_loader, val_loader, device, 
                 learning_rate=1e-3):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        self.criterion = nn.BCELoss()  # Changed from BCEWithLogitsLoss since we have Sigmoid in model
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=30,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        self.best_val_loss = float('inf')
        self.output_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        predictions = []
        labels = []
        
        logger.info(f"\nStarting training epoch...")
        logger.info(f"Number of batches: {len(self.train_loader)}")
        
        # Use tqdm for progress tracking
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training")):
            try:
                if batch_idx % 20 == 0:  # Reduced logging frequency to speed up
                    logger.info(f"Processing batch {batch_idx + 1}/{len(self.train_loader)}")
                
                images = batch['image'].to(self.device, non_blocking=True)
                targets = batch['label'].to(self.device, non_blocking=True)
                
                if batch_idx == 0:
                    logger.info(f"Input shape: {images.shape}")
                    logger.info(f"Target shape: {targets.shape}")
                    logger.info(f"Sample target values: {targets[0]}")
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                
                # Track metrics
                total_loss += loss.item()
                # No need for sigmoid since it's in the model
                predictions.extend(outputs.detach().cpu().numpy())
                labels.extend(targets.cpu().numpy())
                
                if batch_idx % 50 == 0:  # Reduced logging frequency
                    logger.info(f"Current loss: {loss.item():.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}")
                
                # Clear GPU memory between batches
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        avg_loss = total_loss / len(self.train_loader)
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Debug information
        logger.info("\nDebug Information:")
        logger.info(f"Predictions shape: {predictions.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        
        f1 = self._calculate_f1_score(predictions, labels)
        logger.info(f"Calculated F1 score: {f1:.4f}")
        
        return avg_loss, f1

    def _calculate_f1_score(self, predictions, labels, threshold=0.5):
        # Convert probabilities to binary predictions
        binary_predictions = (predictions > threshold).astype(int)
        
        # Calculate F1 score for each class
        f1_scores = []
        for i in range(labels.shape[1]):
            class_f1 = f1_score(labels[:, i], binary_predictions[:, i], zero_division=1)
            f1_scores.append(class_f1)
            logger.info(f"F1 score for class {i}: {class_f1:.4f}")
        
        # Return macro average
        return np.mean(f1_scores)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        predictions = []
        labels = []
        
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validating')):
            try:
                images = batch['image'].to(self.device, non_blocking=True)
                targets = batch['label'].to(self.device, non_blocking=True)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                labels.extend(targets.cpu().numpy())
                
                # Clear GPU memory periodically
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                continue
        
        avg_loss = total_loss / len(self.val_loader)
        f1 = self._calculate_f1_score(np.array(predictions), np.array(labels))
        
        return avg_loss, f1

    def train(self, num_epochs=20, patience=5):  # Reduced default epochs
        logger.info("\nStarting training...")
        patience_counter = 0
        
        # Initialize metrics tracking
        train_losses = []
        train_f1s = []
        val_losses = []
        val_f1s = []
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training phase
            train_loss, train_f1 = self.train_epoch()
            train_losses.append(train_loss)
            train_f1s.append(train_f1)
            
            # Clear memory before validation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Validation phase
            val_loss, val_f1 = self.validate()
            val_losses.append(val_loss)
            val_f1s.append(val_f1)
            
            # Log metrics
            logger.info(
                f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}\n"
                f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}"
            )
            
            # Save best model
            if val_loss < self.best_val_loss:
                logger.info("Saving best model...")
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_f1': val_f1
                }, os.path.join(self.output_dir, 'best_model.pth'))
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Clear memory after each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Plot training metrics
        self._plot_metrics(train_losses, train_f1s, val_losses, val_f1s)
        
        logger.info("Training completed!")

    def _plot_metrics(self, train_losses, train_f1s, val_losses, val_f1s):
        """Plot training and validation metrics"""
        plt.figure(figsize=(12, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Loss over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot F1 scores
        plt.subplot(1, 2, 2)
        plt.plot(train_f1s, label='Train F1')
        plt.plot(val_f1s, label='Val F1')
        plt.title('F1 Score over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_metrics.png'))
        plt.close()

def main():
    try:
        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # GPU setup and memory management
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            torch.cuda.empty_cache()
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU")
        
        # Efficient transform pipeline
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

        logger.info("Loading dataset...")
        dataset = PrescriptionDataset(
            image_dir="dataset/train/resized_images",
            cache_dir="ocr_cache",
            transform=transform,
            force_reload=True,       # Set to True to regenerate labels
            batch_process=True,      # Process in batches to manage memory
            batch_size=20,           # Smaller batch size for more frequent GC
            image_size=(400, 600)    # Reduced image size for OCR
        )
        logger.info(f"Dataset loaded successfully. Total samples: {len(dataset)}")

        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        logger.info(f"Train size: {train_size}, Val size: {val_size}")

        # Configure data loaders with memory optimization
        train_loader = DataLoader(
            train_dataset,
            batch_size=16,          # Reduced batch size
            shuffle=True,
            num_workers=0,          # Avoid additional processes
            pin_memory=True         # Faster data transfer to GPU
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=16,          # Reduced batch size
            shuffle=False,
            num_workers=0,          # Avoid additional processes
            pin_memory=True         # Faster data transfer to GPU
        )

        logger.info("Initializing model...")
        model = MediRemModel().to(device)
        
        # Count model parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model initialized successfully with {total_params/1e6:.2f}M parameters")

        logger.info("Initializing trainer...")
        trainer = PrescriptionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=5e-4      # Slightly reduced learning rate
        )
        logger.info("Trainer initialized successfully")

        logger.info("Starting training...")
        trainer.train(num_epochs=20, patience=5)  # Reduced epochs for faster training
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
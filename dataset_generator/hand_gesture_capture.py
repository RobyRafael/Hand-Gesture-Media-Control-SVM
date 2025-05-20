import cv2
import os
from datetime import datetime
import time
import numpy as np

class HandGestureCapture:
    def __init__(self, max_images_per_gesture=1000, frame_size=(640, 480)):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
        
        # Set base path for datasets
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
        
        # Configuration
        self.max_images = max_images_per_gesture
        self.capture_delay = 0.5  # Changed from 0.1 to 0.5 seconds between captures
        self.frame_size = frame_size
        
        # Initialize state variables
        self.current_label = None
        self.key_pressed = False
        self.last_capture_time = 0
        
        # Ensure dataset directories exist and initialize counters
        self.image_counter = {}
        self.initialize_directories()
        
    def initialize_directories(self):
        """Initialize directory structure and count existing images"""
        for i in range(6):
            dir_path = os.path.join(self.base_path, str(i))
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            self.image_counter[i] = len(os.listdir(dir_path))
            
    def preprocess_frame(self, frame):
        """Preprocess frame for consistency"""
        # Only flip horizontally for mirror effect, no color changes
        frame = cv2.flip(frame, 1)
        return frame, frame  # Return same frame for both display and save
    
    def can_capture(self, label):
        """Check if we can capture more images for this label"""
        return self.image_counter[label] < self.max_images
    
    def capture_and_save(self, label):
        """Capture image from camera and save it with appropriate label"""
        if not self.can_capture(label):
            print(f"Maximum number of images ({self.max_images}) reached for gesture {label}")
            return False
            
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture image")
            return False
            
        # Preprocess the frame
        _, processed_frame = self.preprocess_frame(frame)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"gesture_{label}_{timestamp}.jpg"
        save_path = os.path.join(self.base_path, str(label), filename)
        
        # Save the processed image
        try:
            cv2.imwrite(save_path, processed_frame)
            self.image_counter[label] += 1
            print(f"Captured image for gesture {label}: {filename}")
            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
    
    def draw_interface(self, frame):
        """Draw user interface elements on the frame"""
        height, width = frame.shape[:2]
        
        # Draw guide rectangle
        rect_color = (0, 255, 0)
        cv2.rectangle(frame, (width//4, height//4), (3*width//4, 3*height//4), rect_color, 2)
        
        # Draw gesture counters
        y_offset = 30
        for i in range(6):
            count_text = f"Gesture {i}: {self.image_counter[i]}/{self.max_images}"
            color = (0, 255, 0) if self.current_label == i else (255, 255, 255)
            cv2.putText(frame, count_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
        
        # Draw instructions
        cv2.putText(frame, "Press 0-5 to select gesture label", (10, height - 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Hold key to capture continuously", (10, height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Press ESC to exit", (10, height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main capture loop"""
        print("\n=== Hand Gesture Dataset Capture ===")
        print(f"Maximum images per gesture: {self.max_images}")
        print("Instructions:")
        print("- Press keys 0-5 to select the gesture label")
        print("- Hold the key to continuously capture images")
        print("- Release the key to stop capturing")
        print("- Press ESC to exit")
        print("===================================\n")
        
        try:
            while True:
                # Read and process frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Error reading from camera")
                    break
                
                # Process frame
                display_frame, _ = self.preprocess_frame(frame)
                
                # Draw interface
                display_frame = self.draw_interface(display_frame)
                
                # Display the frame
                cv2.imshow('Hand Gesture Capture', display_frame)
                
                # Handle key input
                key = cv2.waitKey(1) & 0xFF  # Add mask to handle key properly
                
                # Check for ESC key
                if key == 27:  # ESC
                    break
                
                # Handle digit keys (0-5)
                if 48 <= key <= 53:  # ASCII for 0-5
                    digit = key - 48
                    self.current_label = digit
                    
                    # Always capture when a key is pressed, regardless if same as before
                    if self.can_capture(digit):
                        self.capture_and_save(digit)
                        self.last_capture_time = time.time()
                        self.key_pressed = True
                
                # Handle continuous capture
                elif self.key_pressed and self.current_label is not None:
                    # Check if the current key is still the same as the original key
                    if key == (self.current_label + 48):  # Convert back to ASCII
                        current_time = time.time()
                        if (current_time - self.last_capture_time >= self.capture_delay and 
                            self.can_capture(self.current_label)):
                            self.capture_and_save(self.current_label)
                            self.last_capture_time = current_time
                    else:
                        # Key was released or different key pressed
                        self.key_pressed = False
        
        except KeyboardInterrupt:
            print("\nCapture interrupted by user")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
            print("\nCapture session ended")
            print("\nFinal image counts:")
            for i in range(6):
                print(f"Gesture {i}: {self.image_counter[i]} images")

if __name__ == "__main__":
    capture = HandGestureCapture()
    capture.run()
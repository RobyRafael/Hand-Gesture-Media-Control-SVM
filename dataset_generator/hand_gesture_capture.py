import cv2
import os
from datetime import datetime
import time

class HandGestureCapture:
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        
        # Set base path for datasets
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
        
        # Ensure all directories exist
        for i in range(6):  # Create directories 0-5 if they don't exist
            dir_path = os.path.join(self.base_path, str(i))
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
          # Variables for capturing
        self.current_label = None
        self.key_pressed = False
        self.capture_delay = 0.1  # seconds between captures when key is held down (burst mode)
        self.last_capture_time = 0
        
        # Counter for each label
        self.image_counter = {}
        for i in range(6):
            self.image_counter[i] = len(os.listdir(os.path.join(self.base_path, str(i))))
    
    def capture_and_save(self, label):
        """Capture image from camera and save it with appropriate label"""
        # Read frame from camera
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture image")
            return
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Create filename with format: gesture_[label]_[timestamp].jpg
        filename = f"gesture_{label}_{timestamp}.jpg"
        save_path = os.path.join(self.base_path, str(label), filename)
        
        # Save the image
        cv2.imwrite(save_path, frame)
        
        # Update counter
        self.image_counter[label] += 1
        print(f"Captured image for gesture {label}: {filename}")
    
    def run(self):
        print("\n=== Hand Gesture Dataset Capture ===")
        print("Instructions:")
        print("- Press keys 0-5 to select the gesture label")
        print("- Hold the key to continuously capture images")
        print("- Release the key to capture one final image")
        print("- Press ESC to exit")
        print("===================================\n")
        
        try:
            while True:
                # Read frame from camera
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Flip the frame horizontally (mirror view)
                frame = cv2.flip(frame, 1)
                
                # Display label and counter on frame
                if self.current_label is not None:
                    label_text = f"Current label: {self.current_label} | Images: {self.image_counter[self.current_label]}"
                    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add guide rectangle for hand placement
                height, width = frame.shape[:2]
                cv2.rectangle(frame, (width//4, height//4), (3*width//4, 3*height//4), (0, 255, 0), 2)
                
                # Add instructions on the frame
                cv2.putText(frame, "Press 0-5 to select gesture label", (10, height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, "Hold key to capture continuously", (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, "Press ESC to exit", (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display the frame
                cv2.imshow('Hand Gesture Capture', frame)
                
                # Key handling
                key = cv2.waitKey(1)
                
                # Check for ESC key
                if key == 27:  # ESC key
                    break
                
                # Check for digit keys 0-5
                if key >= 48 and key <= 53:  # ASCII for 0-5
                    digit = key - 48
                    
                    if self.current_label != digit:
                        print(f"Selected label: {digit}")
                        self.current_label = digit
                        self.key_pressed = True
                        
                        # Capture immediately when a new label is selected
                        self.capture_and_save(self.current_label)
                        self.last_capture_time = time.time()
                    
                # Process continuous capture if key is held
                if self.key_pressed and self.current_label is not None:
                    current_time = time.time()
                    if current_time - self.last_capture_time >= self.capture_delay:
                        self.capture_and_save(self.current_label)
                        self.last_capture_time = current_time
                
                # Reset key status if no key is pressed
                if key == -1 and self.key_pressed and self.current_label is not None:
                    # Capture one more image when releasing the key
                    self.capture_and_save(self.current_label)
                    print(f"Key released for label {self.current_label}, captured final image")
                    self.key_pressed = False
        
        finally:
            # Clean up
            self.cap.release()
            cv2.destroyAllWindows()
            print("\nCapture session ended")
            for i in range(6):
                count = self.image_counter[i]
                print(f"Label {i}: {count} images")

if __name__ == "__main__":
    # Run the program
    capture = HandGestureCapture()
    capture.run()
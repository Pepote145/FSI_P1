import cv2
import numpy as np

class CarExtractor:

    def __init__(self, video_path, fondo_path):
        self.video_path = video_path
        self.fondo_path = fondo_path

        # Cargar el fondo previamente calculado
        self.fondo = cv2.imread(self.fondo_path, cv2.IMREAD_GRAYSCALE)
        self.cap = cv2.VideoCapture(self.video_path)

    def subtract_background(self):
        # Calcular la diferencia absoluta entre el frame y el fondo
        diff = cv2.absdiff(gray_frame, self.fondo)
        return diff

    def binanyzation(self, image, threshold=30):
        # Aplicar un umbral para binarizar la imagen
        _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return binary_image

    def morphological_operations(self, binary_image, kernel_size=4):
        # Crear un kernel para las operaciones morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        # Aplicar operaciones morfológicas para limpiar la imagen binaria
        cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_CLOSE, kernel)
        
        #dilatar la imagen para resaltar mejor los objetos
        cleaned_image = cv2.dilate(cleaned_image, kernel, iterations=2)

        return cleaned_image
    
if __name__ == "__main__":
    video_path = "autopista.mp4"
    fondo_path = "fondo_autopista.jpg"
    extractor = CarExtractor(video_path, fondo_path)
    while True:
        ret, frame = extractor.cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resta de fondo
        diff = extractor.subtract_background()

        # Binarización
        binary_image = extractor.binanyzation(diff)

        # Operaciones morfológicas
        morfologic = extractor.morphological_operations(binary_image)

        # Mostrar resultados (opcional)
        cv2.imshow("frame original", frame)
        cv2.imshow("diferencia con fondo", diff)
        cv2.imshow("imagen binaria", binary_image)
        cv2.imshow("imagen morfológica", morfologic)

        if cv2.waitKey(30) & 0xFF == 27:  # Salir con 'Esc'
            break
    extractor.cap.release()
    cv2.destroyAllWindows()



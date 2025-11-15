import cv2
import numpy as np

# 6.- Detección de contornos
class VideoContourDetection:
    def __init__(self, video_path, fondo_path, threshold=60):
        self.video_path = video_path
        self.fondo_path = fondo_path
        self.threshold = threshold

    def process_video(self):
        # Cargar el fondo en escala de grises
        fondo = cv2.imread(self.fondo_path, cv2.IMREAD_GRAYSCALE)

        # Abrir el video
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print("ERROR: No se pudo abrir el video. Revisa la ruta del archivo.")
            return

        # Definir el kernel para operaciones morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convertir el frame actual a escala de grises
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calcular la diferencia absoluta entre el frame y el fondo
            diff = cv2.absdiff(gray_frame, fondo)

            # Aplicar umbralización para binarizar la imagen
            _, binary_image = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

            # Aplicar operaciones morfológicas para limpiar la imagen
            morphed_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
            morphed_image = cv2.morphologyEx(morphed_image, cv2.MORPH_CLOSE, kernel)
            morphed_image = cv2.dilate(morphed_image, kernel, iterations=2)

            # Detectar contornos
            contours, _ = cv2.findContours(morphed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Dibujar contornos con un rectangulo en el frame original 

            contour_frame = frame.copy()
            for contour in contours:
                if cv2.contourArea(contour) < 2000:  # Filtrar contornos pequeños
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(contour_frame, (x, y), (x + w, y + h), (0, 255, 0), 4)


            # Mostrar la imagen con contornos
            cv2.imshow("Contornos Detectados (Coches en movimiento)", contour_frame)

            # Salir con la tecla 'q'
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    # Ruta relativa correcta
    video_path = "autopista.mp4"
    fondo_path = "fondo_autopista.jpg"
    threshold_value = 30  # Ajusta este valor según sea necesario
    detector = VideoContourDetection(video_path, fondo_path, threshold_value)
    detector.process_video()

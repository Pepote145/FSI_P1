import cv2
import numpy as np

class BackgroundExtractor:
    def __init__(self, video_path, alpha=0.01, use_gray=True, frame_skip=2):
        self.video_path = video_path
        self.alpha = alpha
        self.use_gray = use_gray
        self.frame_skip = max(1, int(frame_skip))

    def extract_background(self):
        cap = cv2.VideoCapture(self.video_path)

        # Leer primer frame correctamente
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError("No se pudo leer el primer frame del video.")

        # Convertir a gris si se pidió
        if self.use_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Acumulador en float32
        avg = np.float32(frame)

        # Procesar frames restantes
        while True:
            # Saltar frames para acelerar (si frame_skip > 1)
            for _ in range(self.frame_skip):
                ret, frame = cap.read()
                if not ret:
                    break
            if not ret:
                break

            # Gris si corresponde
            if self.use_gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Actualizar fondo incremental
            cv2.accumulateWeighted(frame, avg, self.alpha)

        cap.release()

        # Convertir el acumulador en imagen final
        background = cv2.convertScaleAbs(avg)

        # Guardado opcional
        if self.use_gray:
            cv2.imwrite("fondo_autopista_gris.jpg", background)
        cv2.imwrite("fondo_autopista.jpg", background)

        return background


if __name__ == "__main__":
    video_path = "autopista.mp4"
    extractor = BackgroundExtractor(video_path)
    background = extractor.extract_background()

    cv2.imshow("Fondo de la Autopista", background)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

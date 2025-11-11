import cv2
import numpy as np
import matplotlib.pyplot as plt # Plotting library

class OnlyBackGround:
    def __init__(self, video):
        self.video = video
        self.background = None

    def chargevideo(self):
        cap = cv2.VideoCapture(self.video)

        # Comprobamos si el video se pudo abrir
        if not cap.isOpened():
            print("ERROR: No se pudo abrir el video. Revisa la ruta del archivo.")
            return

        frames = []
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames.append(frame)
            count += 1

        cap.release()
        print(f"Frames cargados: {count}")

        if count == 0:
            print("⚠️ No se leyeron frames. Posiblemente el video esté vacío o la ruta sea incorrecta.")
            return

        # Convertir lista a array y calcular el fondo
        frames = np.array(frames, dtype=np.uint8)
        self.background = np.median(frames, axis=0).astype(np.uint8)

        # Guardar y mostrar la imagen del fondo
        cv2.imwrite("fondo_autopista.jpg", self.background)
        cv2.imshow("Fondo autopista", self.background)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Ruta relativa correcta
    video_path = "Primera_practica/autopista.mp4"
    bg = OnlyBackGround(video_path)
    bg.chargevideo()

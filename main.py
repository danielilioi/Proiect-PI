import torch
import cv2
import pyzed.sl as sl
import pathlib
import tkinter as tk
from tkinter import filedialog, simpledialog
import sys


path_temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


def procesareFrame(frame, modelYolo):
    rezultat = modelYolo(frame)
    detectie = rezultat.xyxy[0]
    for *box, conf, cls in detectie:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame


def zedYolo(model, salvareBool=False, pathSvoSalvare=None):
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30

    functioneareZed = zed.open(init_params)
    if functioneareZed != sl.ERROR_CODE.SUCCESS:
        print("Nu s-a putut deschide camera ZED")
        exit()

    if salvareBool and pathSvoSalvare:
        parametriDeInregistrare = sl.RecordingParameters(pathSvoSalvare, sl.SVO_COMPRESSION_MODE.H264)
        err = zed.enable_recording(parametriDeInregistrare)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Eroare la inregistrarea SVO: {err}")
            exit()
        else:
            print(f"Inregistrare SVO pornita: {pathSvoSalvare}")

    parametriRuntime = sl.RuntimeParameters()
    imagine = sl.Mat()

    try:
        while True:
            if zed.grab(parametriRuntime) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(imagine, sl.VIEW.LEFT)
                frame = imagine.get_data()
                frame = procesareFrame(frame, model)
                cv2.imshow('Detecții Zed', frame)

                if salvareBool:
                    zed.record()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        if salvareBool:
            zed.disable_recording()
            print("Inregistrare Svo oprita.")
        zed.close()
        cv2.destroyAllWindows()


def videoYolo(model, video_path, pathDeSalvare=None):
    captura = cv2.VideoCapture(video_path)
    rezultat = None

    if pathDeSalvare:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        rezultat = cv2.VideoWriter(pathDeSalvare, fourcc, 20.0, (int(captura.get(3)), int(captura.get(4))))

    while captura.isOpened():
        ret, frame = captura.read()
        if not ret:
            break
        frame = procesareFrame(frame, model)
        cv2.imshow('Detectie Video', frame)
        if rezultat:
            rezultat.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    captura.release()
    if rezultat:
        rezultat.release()
    cv2.destroyAllWindows()

def imagineYolo(model, patPtImagine, pathDeSalvare=None):
    frame = cv2.imread(patPtImagine)
    frame = procesareFrame(frame, model)
    if pathDeSalvare:
        cv2.imwrite(pathDeSalvare, frame)
    frame = cv2.resize(frame,(900,600))
    cv2.imshow('Detecții Imagine', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def selectareTipFisier(tip_fisier):
    root = tk.Tk()
    root.withdraw()
    if tip_fisier == 'video':
        cale = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")])
    else:
        cale = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    return cale


def salvare(tip_fisier):
    root = tk.Tk()
    root.withdraw()
    if tip_fisier == 'video':
        path_de_salvare = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
    elif tip_fisier == 'svo':
        path_de_salvare = filedialog.asksaveasfilename(defaultextension=".svo", filetypes=[("SVO files", "*.svo")])
    else:
        path_de_salvare = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPG files", "*.jpg"), ("PNG files", "*.png")])
    return path_de_salvare


if __name__ == "__main__":
    model = torch.hub.load('./yolov5', 'custom', path='./yolov5/runs/train/exp/weights/best.pt', source='local',force_reload=True)

    print("Selectați o opțiune:")
    print("1: Camera ZED")
    print("2: Video")
    print("3: Imagine")
    alegere = input("Alegerea dvs. (1/2/3): ")

    optiuneSalvare = simpledialog.askstring("Salvare", "Doriți să salvați rezultatele? (da/nu):").lower() == 'da'
    path_de_salvare = None
    
    if optiuneSalvare:
        if alegere == '2':
            path_de_salvare = salvare('video')
        elif alegere == '3':
            path_de_salvare = salvare('image')
        elif alegere == '1':
            path_de_salvare = salvare('svo')

    if alegere == '1':
        zedYolo(model, salvareBool=optiuneSalvare, pathSvoSalvare=path_de_salvare)
    elif alegere == '2':
        pathVideo = selectareTipFisier('video')
        if pathVideo:
            videoYolo(model, pathVideo, path_de_salvare)
        else:
            print("Nu s-a selectat niciun fișier video.")
    elif alegere == '3':
        pathImagine = selectareTipFisier('image')
        if pathImagine:
            imagineYolo(model, pathImagine, path_de_salvare)
        else:
            print("Nu s-a selectat nicio imagine.")
    else:
        print("Opțiune invalidă!")
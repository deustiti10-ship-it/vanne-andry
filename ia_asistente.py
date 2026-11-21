"""
ASISTENTE DE IA: pregunta lo que quieras y responde con TEXTO + IMAGEN + VOZ/AUDIO + VIDEO.

Dos modos (los dos puntos del trabajo):
  1. IA GENERATIVA   -> un modelo de Machine Learning responde la pregunta completa.
  2. IA DISTRIBUTIVA -> la pregunta (macroproceso) se divide en subprocesos, cada agente
                        resuelve uno y un coordinador junta todo en la respuesta final.
"""
import os
import time
import queue
import threading
import urllib.parse
import requests
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
from gtts import gTTS
from moviepy import ImageClip, AudioFileClip

# ---------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------
MODELO = "llama3.2"
OLLAMA_URL = "http://localhost:11434"

BASE = os.path.dirname(os.path.abspath(__file__))
CARPETA = os.path.join(BASE, "salida_asistente")
os.makedirs(CARPETA, exist_ok=True)
IMAGEN = os.path.join(CARPETA, "imagen.png")
VOZ = os.path.join(CARPETA, "voz.mp3")
VIDEO = os.path.join(CARPETA, "video.mp4")


# ---------------------------------------------------------------
# OLLAMA
# ---------------------------------------------------------------
def elegir_modelo():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        r.raise_for_status()
    except Exception:
        raise RuntimeError("No se pudo conectar con Ollama. Abre la aplicacion Ollama y reintenta.")
    instalados = [m["name"] for m in r.json().get("models", [])]
    if not instalados:
        raise RuntimeError(f"Ollama no tiene modelos. Ejecuta: ollama pull {MODELO}")
    for nombre in instalados:
        if nombre.split(":")[0] == MODELO.split(":")[0]:
            return nombre
    return instalados[0]


def llamar_ollama(modelo, prompt):
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": modelo, "prompt": prompt, "stream": False},
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["response"].strip()


# ---------------------------------------------------------------
# RESPUESTA: MODO GENERATIVO Y MODO DISTRIBUTIVO
# ---------------------------------------------------------------
def responder_generativo(modelo, pregunta, emitir):
    emitir("log", "[IA GENERATIVA] Un solo modelo responde la pregunta completa...")
    return llamar_ollama(
        modelo,
        f"Responde siempre en español, de forma clara y en máximo 120 palabras, "
        f"sin viñetas ni asteriscos: {pregunta}",
    )


def dividir_en_subprocesos(modelo, pregunta):
    """PLANEACION: el planificador divide el macroproceso en 3 o 4 subprocesos."""
    texto = llamar_ollama(
        modelo,
        "Divide esta tarea o pregunta en 3 o 4 subprocesos independientes, cada uno como una "
        "pregunta corta en español. Escribe un subproceso por línea, sin explicaciones.\n"
        f"Tarea: {pregunta}",
    )
    lineas = []
    for l in texto.splitlines():
        l = l.strip().lstrip("-*•0123456789.) ").strip()
        if len(l) > 8:
            lineas.append(l)
    return lineas[:4] or [pregunta]


def responder_distributivo(modelo, pregunta, emitir):
    emitir("log", "[IA DISTRIBUTIVA] Planeación: dividiendo el macroproceso en subprocesos...")
    subprocesos = dividir_en_subprocesos(modelo, pregunta)
    emitir("log", f"Macroproceso: {pregunta}")
    for i, s in enumerate(subprocesos, 1):
        emitir("log", f"  Subproceso {i}: {s}")
    emitir("log", "")

    informes = []
    for i, s in enumerate(subprocesos, 1):
        emitir("log", f"Agente {i} trabajando...")
        r = llamar_ollama(
            modelo,
            f"Eres el agente {i}, encargado solo de este subproceso: {s}\n"
            f"(Contexto general: {pregunta})\n"
            "Responde en español, en máximo 50 palabras, sin viñetas ni asteriscos.",
        )
        informes.append(f"Subproceso {i} ({s}): {r}")
        emitir("log", f"--- Agente {i} ---\n{r}\n")

    emitir("log", "Coordinador: integrando los informes de los agentes...")
    return llamar_ollama(
        modelo,
        f"Eres el coordinador. Tarea original: {pregunta}\n\nInformes de los agentes:\n"
        + "\n".join(informes)
        + "\n\nEscribe en español una respuesta final única de máximo 120 palabras, "
        "sin viñetas ni asteriscos, que integre todo.",
    )


# ---------------------------------------------------------------
# IMAGEN, VOZ Y VIDEO
# ---------------------------------------------------------------
def crear_imagen(modelo, pregunta, respuesta, emitir):
    desc = llamar_ollama(
        modelo,
        "Write, in English, one single line of maximum 25 words describing a visual scene "
        "that illustrates this answer, for an image generator. Return only the description.\n"
        f"Question: {pregunta}\nAnswer: {respuesta}",
    ).splitlines()[0].strip().strip('"')
    emitir("log", f"Descripción de la imagen: {desc}")
    url = (
        "https://image.pollinations.ai/prompt/" + urllib.parse.quote(desc)
        + "?width=1024&height=576&nologo=true"
    )
    for intento in range(6):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            with open(IMAGEN, "wb") as f:
                f.write(r.content)
            return
        except Exception:
            emitir("log", f"  Error con la imagen, reintentando ({intento + 1}/6)...")
            time.sleep(8)
    raise RuntimeError("No se pudo generar la imagen. Revisa tu internet.")


def crear_voz_y_video(respuesta):
    limpio = respuesta.replace("*", "").replace("#", "")
    gTTS(text=limpio, lang="es").save(VOZ)
    audio = AudioFileClip(VOZ)
    clip = ImageClip(IMAGEN).with_duration(audio.duration).with_audio(audio)
    clip.write_videofile(VIDEO, fps=24, codec="libx264", audio_codec="aac", logger=None)


def procesar(pregunta, modo, emitir):
    modelo = elegir_modelo()
    emitir("log", f"Modelo de Ollama: {modelo}\n")
    if modo == "distributivo":
        respuesta = responder_distributivo(modelo, pregunta, emitir)
    else:
        respuesta = responder_generativo(modelo, pregunta, emitir)
    emitir("log", f"\n=== RESPUESTA ===\n{respuesta}\n")

    emitir("log", "Generando imagen...")
    crear_imagen(modelo, pregunta, respuesta, emitir)
    emitir("imagen", IMAGEN)

    emitir("log", "Generando voz y video...")
    crear_voz_y_video(respuesta)
    emitir("log", "\n¡Listo! Ya puedes oír la voz o ver el video.")
    emitir("listo", "")


# ---------------------------------------------------------------
# VENTANA UNICA
# ---------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.cola = queue.Queue()
        self.foto = None
        root.title("Asistente de IA: texto, imagen, voz y video")
        root.geometry("1150x700")

        tk.Label(root, text="ASISTENTE DE IA - Pregúntame lo que quieras",
                 font=("Segoe UI", 15, "bold")).pack(pady=(10, 4))

        fila = tk.Frame(root)
        fila.pack(fill="x", padx=12)
        tk.Label(fila, text="Pregunta:").pack(side="left")
        self.entrada = tk.Entry(fila, font=("Segoe UI", 11))
        self.entrada.insert(0, "¿Cuál es el país que más páramos tiene en el mundo?")
        self.entrada.pack(side="left", fill="x", expand=True, padx=8)
        self.entrada.bind("<Return>", lambda e: self.iniciar("generativo"))
        self.b_gen = tk.Button(fila, text="Responder (IA generativa)", width=24,
                               command=lambda: self.iniciar("generativo"))
        self.b_gen.pack(side="left", padx=2)
        self.b_dis = tk.Button(fila, text="Resolver por subprocesos (distributiva)", width=34,
                               command=lambda: self.iniciar("distributivo"))
        self.b_dis.pack(side="left", padx=2)

        centro = tk.Frame(root)
        centro.pack(fill="both", expand=True, padx=12, pady=10)
        self.texto = scrolledtext.ScrolledText(centro, wrap="word", font=("Consolas", 10), width=70)
        self.texto.pack(side="left", fill="both", expand=True)

        der = tk.Frame(centro)
        der.pack(side="left", fill="y", padx=(10, 0))
        self.lbl_imagen = tk.Label(der, text="(aquí aparecerá la imagen)", width=60, height=15,
                                   relief="sunken")
        self.lbl_imagen.pack()
        botones = tk.Frame(der)
        botones.pack(pady=10)
        self.b_audio = tk.Button(botones, text="Reproducir voz/audio", state="disabled",
                                 command=lambda: os.startfile(VOZ))
        self.b_audio.pack(side="left", padx=4)
        self.b_video = tk.Button(botones, text="Ver video", state="disabled",
                                 command=lambda: os.startfile(VIDEO))
        self.b_video.pack(side="left", padx=4)
        tk.Button(botones, text="Abrir carpeta",
                  command=lambda: os.startfile(CARPETA)).pack(side="left", padx=4)

        self.root.after(150, self.revisar_cola)

    def emitir(self, tipo, dato):
        self.cola.put((tipo, dato))

    def activar(self, estado):
        for b in (self.b_gen, self.b_dis):
            b.config(state=estado)

    def iniciar(self, modo):
        pregunta = self.entrada.get().strip()
        if not pregunta or str(self.b_gen["state"]) == "disabled":
            return
        self.activar("disabled")
        self.b_audio.config(state="disabled")
        self.b_video.config(state="disabled")
        self.texto.delete("1.0", "end")
        self.texto.insert("end", f"Pregunta: {pregunta}\n\n")

        def trabajo():
            try:
                procesar(pregunta, modo, self.emitir)
            except Exception as e:
                self.emitir("log", f"\nERROR: {e}")
                self.emitir("fin", "")

        threading.Thread(target=trabajo, daemon=True).start()

    def revisar_cola(self):
        try:
            while True:
                tipo, dato = self.cola.get_nowait()
                if tipo == "log":
                    self.texto.insert("end", dato + "\n")
                    self.texto.see("end")
                elif tipo == "imagen":
                    img = Image.open(dato)
                    img.thumbnail((480, 270))
                    self.foto = ImageTk.PhotoImage(img)
                    self.lbl_imagen.config(image=self.foto, text="", width=img.width, height=img.height)
                elif tipo == "listo":
                    self.b_audio.config(state="normal")
                    self.b_video.config(state="normal")
                    self.activar("normal")
                elif tipo == "fin":
                    self.activar("normal")
        except queue.Empty:
            pass
        self.root.after(150, self.revisar_cola)


if __name__ == "__main__":
    ventana = tk.Tk()
    App(ventana)
    ventana.mainloop()

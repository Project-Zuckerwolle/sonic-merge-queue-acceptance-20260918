"""Nova Predator v1 — Design-Tool (ehemals Layer 5).

Generiert vollständige UI-Code-Dateien für Apex.
Wird als design_ui Tool von react_brain aufgerufen.
"""
from __future__ import annotations

from core.logger import get

log = get("layer6.design_tool")


def _dark_palette():
    return {"bg": "#1a1a2e", "fg": "#e8e8f0", "accent": "#00d4ff",
            "btn_bg": "#16213e", "border": "#252540"}


def _light_palette():
    return {"bg": "#f5f5f5", "fg": "#1a1a1a", "accent": "#0066cc",
            "btn_bg": "#e0e8f0", "border": "#d0d0d0"}


def _palette(stil: str) -> dict:
    return _dark_palette() if "dark" in stil else _light_palette()


def gen_tkinter(beschreibung: str, stil: str, komponenten: list[str]) -> str:
    """Tkinter-App mit sauberem Layout aus Komponenten-Liste."""
    p = _palette(stil)

    widget_lines: list[str] = []
    method_lines: list[str] = []
    row = 0

    for komp in (komponenten or ["button:Starten", "label:Status"]):
        parts = komp.split(":", 1)
        typ   = parts[0].strip().lower()
        label = parts[1].strip() if len(parts) > 1 else typ.capitalize()
        safe  = label.lower().replace(" ", "_").replace("-", "_")

        if typ == "button":
            widget_lines.append(
                f'        tk.Button(self.frame, text="{label}", '
                f'bg="{p["accent"]}", fg="{p["bg"]}", activebackground="{p["fg"]}", '
                f'font=self.font_btn, relief="flat", pady=6, '
                f'command=self.on_{safe}).grid(row={row}, column=0, '
                f'columnspan=2, sticky="ew", pady=4)'
            )
            method_lines.append(
                f'    def on_{safe}(self):\n'
                f'        """Handler für Button: {label}"""\n'
                f'        self.set_status("{label} gedrückt")\n'
                f'        # TODO: Logik implementieren'
            )

        elif typ == "label":
            var = f"self.lbl_{safe}_var"
            widget_lines.append(f'        {var} = tk.StringVar(value="{label}")')
            widget_lines.append(
                f'        tk.Label(self.frame, textvariable={var}, '
                f'bg="{p["bg"]}", fg="{p["fg"]}", font=self.font'
                f').grid(row={row}, column=0, sticky="w", pady=2)'
            )

        elif typ in ("entry", "input"):
            var = f"self.entry_{safe}_var"
            widget_lines.append(f'        {var} = tk.StringVar()')
            widget_lines.append(
                f'        tk.Label(self.frame, text="{label}:", '
                f'bg="{p["bg"]}", fg="{p["fg"]}", font=self.font'
                f').grid(row={row}, column=0, sticky="w")'
            )
            widget_lines.append(
                f'        tk.Entry(self.frame, textvariable={var}, '
                f'bg="{p["btn_bg"]}", fg="{p["fg"]}", insertbackground="{p["fg"]}", '
                f'font=self.font, relief="flat"'
                f').grid(row={row}, column=1, sticky="ew", pady=2, padx=(4,0))'
            )

        elif typ in ("listbox", "list"):
            var = f"self.lb_{safe}"
            widget_lines.append(
                f'        {var} = tk.Listbox(self.frame, bg="{p["btn_bg"]}", '
                f'fg="{p["fg"]}", selectbackground="{p["accent"]}", '
                f'font=self.font, relief="flat", height=10)'
            )
            widget_lines.append(
                f'        {var}.grid(row={row}, column=0, columnspan=2, '
                f'sticky="nsew", pady=4)'
            )
            widget_lines.append(
                f'        tk.Scrollbar(self.frame, command={var}.yview'
                f').grid(row={row}, column=2, sticky="ns")'
            )

        elif typ == "progressbar":
            widget_lines.append(
                f'        self.progress = ttk.Progressbar(self.frame, '
                f'mode="determinate", maximum=100)'
            )
            widget_lines.append(
                f'        self.progress.grid(row={row}, column=0, '
                f'columnspan=2, sticky="ew", pady=4)'
            )

        elif typ in ("text", "textarea"):
            var = f"self.txt_{safe}"
            widget_lines.append(
                f'        {var} = tk.Text(self.frame, bg="{p["btn_bg"]}", '
                f'fg="{p["fg"]}", insertbackground="{p["fg"]}", '
                f'font=("Consolas", 10), relief="flat", height=8, wrap="word")'
            )
            widget_lines.append(
                f'        {var}.grid(row={row}, column=0, columnspan=2, '
                f'sticky="nsew", pady=4)'
            )

        elif typ == "combobox":
            var = f"self.combo_{safe}_var"
            widget_lines.append(f'        {var} = tk.StringVar()')
            widget_lines.append(
                f'        ttk.Combobox(self.frame, textvariable={var}, '
                f'values=["{label}"], state="readonly"'
                f').grid(row={row}, column=0, columnspan=2, sticky="ew", pady=2)'
            )

        elif typ == "checkbox":
            var = f"self.chk_{safe}_var"
            widget_lines.append(f'        {var} = tk.BooleanVar()')
            widget_lines.append(
                f'        tk.Checkbutton(self.frame, text="{label}", '
                f'variable={var}, bg="{p["bg"]}", fg="{p["fg"]}", '
                f'activebackground="{p["bg"]}", selectcolor="{p["btn_bg"]}", '
                f'font=self.font'
                f').grid(row={row}, column=0, sticky="w", pady=2)'
            )

        row += 1

    widgets_block  = "\n".join(widget_lines) if widget_lines else "        pass"
    methods_block  = "\n\n".join(method_lines)
    title_short    = beschreibung[:50]

    return (
        '#!/usr/bin/env python3\n'
        f'"""\n{beschreibung}\nGeneriert von Nova Predator v1 — design_ui\n"""\n'
        'import tkinter as tk\n'
        'from tkinter import ttk, filedialog, messagebox\n'
        'import sys\n'
        '\n'
        '\n'
        'class App(tk.Tk):\n'
        '    def __init__(self):\n'
        '        super().__init__()\n'
        f'        self.title("{title_short}")\n'
        f'        self.configure(bg="{p["bg"]}")\n'
        '        self.resizable(True, True)\n'
        '        self.minsize(520, 380)\n'
        '\n'
        '        self.font     = ("Segoe UI", 10)\n'
        '        self.font_btn = ("Segoe UI", 10, "bold")\n'
        '        self.font_h   = ("Segoe UI", 13, "bold")\n'
        '\n'
        '        # ── Titelzeile\n'
        f'        tk.Label(self, text="{title_short}", bg="{p["bg"]}", '
        f'fg="{p["accent"]}", font=self.font_h, pady=12).pack(fill="x", padx=16)\n'
        '\n'
        '        # ── Trennlinie\n'
        f'        tk.Frame(self, bg="{p["border"]}", height=1).pack(fill="x")\n'
        '\n'
        '        # ── Haupt-Frame\n'
        f'        self.frame = tk.Frame(self, bg="{p["bg"]}", padx=16, pady=12)\n'
        '        self.frame.pack(fill="both", expand=True)\n'
        '        self.frame.columnconfigure(1, weight=1)\n'
        '\n'
        '        # ── Widgets\n'
        f'{widgets_block}\n'
        '\n'
        '        # ── Statusleiste\n'
        '        self.status_var = tk.StringVar(value="Bereit")\n'
        f'        tk.Label(self, textvariable=self.status_var, bg="{p["btn_bg"]}", '
        f'fg="{p["fg"]}", font=("Segoe UI", 9), anchor="w", pady=4, padx=8\n'
        '                 ).pack(fill="x", side="bottom")\n'
        '\n'
        '    def set_status(self, text: str) -> None:\n'
        '        self.status_var.set(text)\n'
        '        self.update_idletasks()\n'
        '\n'
        + (f'{methods_block}\n\n' if methods_block else '')
        + '\n'
        'def main():\n'
        '    app = App()\n'
        '    app.mainloop()\n'
        '\n'
        '\n'
        'if __name__ == "__main__":\n'
        '    main()\n'
    )


def gen_html(beschreibung: str, stil: str, komponenten: list[str]) -> str:
    """HTML/CSS/JS Single-File App."""
    dark   = "dark" in stil
    bg     = "#0a0a0c" if dark else "#f8f9fa"
    fg     = "#e2e2ea" if dark else "#1a1a2e"
    acc    = "#00d4ff" if dark else "#0066cc"
    card   = "#111114" if dark else "#ffffff"
    border = "#252530" if dark else "#dee2e6"
    input_bg = "#18181c" if dark else "#ffffff"

    elements = []
    for komp in (komponenten or []):
        parts = komp.split(":", 1)
        typ   = parts[0].strip().lower()
        label = parts[1].strip() if len(parts) > 1 else typ.capitalize()
        safe  = label.replace(" ", "_").replace("-", "_")

        if typ == "button":
            elements.append(f'  <button class="btn" id="btn_{safe}" onclick="on_{safe}()">{label}</button>')
        elif typ in ("entry", "input"):
            elements.append(f'  <label class="lbl">{label}</label>')
            elements.append(f'  <input type="text" class="input" id="inp_{safe}" placeholder="{label}">')
        elif typ in ("listbox", "list"):
            elements.append(f'  <div class="list-label">{label}</div>')
            elements.append(f'  <select class="select" id="sel_{safe}" size="6" multiple></select>')
        elif typ == "progressbar":
            elements.append(f'  <label class="lbl">{label}</label>')
            elements.append(f'  <progress id="prog_{safe}" class="progress" value="0" max="100"></progress>')
        elif typ in ("text", "textarea"):
            elements.append(f'  <label class="lbl">{label}</label>')
            elements.append(f'  <textarea class="textarea" id="ta_{safe}" rows="6"></textarea>')

    elements_html = "\n".join(elements)
    title_short   = beschreibung[:50]

    return (
        '<!DOCTYPE html>\n'
        '<html lang="de">\n'
        '<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f'<title>{title_short}</title>\n'
        '<style>\n'
        f'*{{box-sizing:border-box;margin:0;padding:0}}\n'
        f'body{{background:{bg};color:{fg};font-family:"Segoe UI",system-ui,sans-serif;'
        f'min-height:100vh;padding:24px;line-height:1.5}}\n'
        f'h1{{color:{acc};font-size:1.4rem;margin-bottom:20px;font-weight:700}}\n'
        f'.card{{background:{card};border:1px solid {border};border-radius:12px;'
        f'padding:24px;max-width:720px;margin:0 auto;display:flex;flex-direction:column;gap:12px}}\n'
        f'.btn{{background:{acc};color:{bg};border:none;padding:10px 20px;border-radius:7px;'
        f'cursor:pointer;font-size:14px;font-weight:600;align-self:flex-start}}\n'
        f'.btn:hover{{opacity:.85}}.btn:active{{transform:scale(.98)}}\n'
        f'.input,.select,.textarea{{width:100%;background:{input_bg};border:1px solid {border};'
        f'color:{fg};padding:9px 12px;border-radius:7px;font-size:14px;font-family:inherit}}\n'
        f'.textarea{{resize:vertical;font-family:"Consolas",monospace;font-size:13px}}\n'
        f'.lbl{{font-size:13px;color:{acc};font-weight:600;margin-bottom:2px}}\n'
        f'.list-label{{font-size:13px;color:{acc};font-weight:600}}\n'
        f'.progress{{width:100%;height:8px;border-radius:4px;border:none;background:{border}}}\n'
        f'.output{{background:{input_bg};border:1px solid {border};border-radius:7px;'
        f'padding:12px;font-family:"Consolas",monospace;font-size:12px;'
        f'white-space:pre-wrap;min-height:80px;color:{fg}}}\n'
        f'.status{{font-size:12px;color:gray}}\n'
        '</style>\n'
        '</head>\n'
        '<body>\n'
        '<div class="card">\n'
        f'  <h1>{title_short}</h1>\n'
        f'{elements_html}\n'
        '  <div class="output" id="output">Bereit...</div>\n'
        '  <div class="status" id="status"></div>\n'
        '</div>\n'
        '<script>\n'
        'function setOutput(t){document.getElementById("output").textContent=t}\n'
        'function setStatus(t){document.getElementById("status").textContent=t}\n'
        '// TODO: Handler implementieren\n'
        '</script>\n'
        '</body>\n'
        '</html>\n'
    )


def gen_customtkinter(beschreibung: str, stil: str) -> str:
    """Customtkinter-Gerüst."""
    theme = "dark" if "dark" in stil else "light"
    title = beschreibung[:50]
    return (
        '#!/usr/bin/env python3\n'
        f'"""\n{beschreibung}\n"""\n'
        'try:\n'
        '    import customtkinter as ctk\n'
        'except ImportError:\n'
        '    import subprocess, sys\n'
        '    subprocess.check_call([sys.executable, "-m", "pip", "install", "customtkinter"])\n'
        '    import customtkinter as ctk\n'
        '\n'
        f'ctk.set_appearance_mode("{theme}")\n'
        'ctk.set_default_color_theme("blue")\n'
        '\n'
        '\n'
        'class App(ctk.CTk):\n'
        '    def __init__(self):\n'
        '        super().__init__()\n'
        f'        self.title("{title}")\n'
        '        self.geometry("640x480")\n'
        '        self._build_ui()\n'
        '\n'
        '    def _build_ui(self):\n'
        f'        ctk.CTkLabel(self, text="{title}", font=("Segoe UI", 16, "bold")).pack(pady=20)\n'
        '        self.btn = ctk.CTkButton(self, text="Starten", command=self.on_starten)\n'
        '        self.btn.pack(pady=10)\n'
        '        self.status = ctk.CTkLabel(self, text="Bereit")\n'
        '        self.status.pack(pady=5)\n'
        '\n'
        '    def on_starten(self):\n'
        '        self.status.configure(text="Läuft...")\n'
        '        # TODO: Logik implementieren\n'
        '\n'
        '\n'
        'if __name__ == "__main__":\n'
        '    App().mainloop()\n'
    )


def generate(framework: str, beschreibung: str, stil: str,
             komponenten: list[str]) -> str:
    """Haupt-Entry: generiert UI-Code für das gewählte Framework."""
    if framework == "html":
        return gen_html(beschreibung, stil, komponenten)
    elif framework == "customtkinter":
        return gen_customtkinter(beschreibung, stil)
    else:
        return gen_tkinter(beschreibung, stil, komponenten)

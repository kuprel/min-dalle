from min_dalle import MinDalle
import sys
import PIL
import PIL.Image
import PIL.ImageTk
import tkinter
from tkinter import ttk

# -- decide stuff --

def regen_root():
    global root
    global blank_image
    global padding_image

    root = tkinter.Tk()
    root.wm_resizable(False, False)

    blank_image = PIL.ImageTk.PhotoImage(PIL.Image.new(size=(256, 256), mode="RGB"))
    padding_image = PIL.ImageTk.PhotoImage(PIL.Image.new(size=(16, 16), mode="RGBA"))

regen_root()

# -- --

meganess = None
def set_mega_true_and_destroy():
	global meganess
	meganess = True
	root.destroy()
def set_mega_false_and_destroy():
	global meganess
	meganess = False
	root.destroy()

frm = ttk.Frame(root, padding=16)
frm.grid()
ttk.Button(frm, text="Mega", command=set_mega_true_and_destroy).grid(column=0, row=0)
ttk.Label(frm, image=padding_image).grid(column=1, row=0)
ttk.Button(frm, text="Not-Mega", command=set_mega_false_and_destroy).grid(column=2, row=0)
root.mainloop()

if meganess is None:
	print("no option selected, goodbye")
	sys.exit(0)

print("confirmed mega: ", str(meganess))

# -- --

model = MinDalle(
    is_mega=meganess, 
    models_root="./pretrained",
    is_reusable=True,
    is_verbose=True
)

# -- --

regen_root()

# -- --

label_image_content = blank_image

sv_prompt = tkinter.StringVar(value="mouse")
sv_temperature = tkinter.StringVar(value="1")
sv_topk = tkinter.StringVar(value="1024")
sv_supercond = tkinter.StringVar(value="16")

def generate():
    # check fields
    try:
        temperature = float(sv_temperature.get())
    except:
        sv_temperature.set("ERROR")
        return
    try:
        topk = int(sv_topk.get())
    except:
        sv_topk.set("ERROR")
        return
    try:
        supercond = int(sv_supercond.get())
    except:
        sv_supercond.set("ERROR")
        return
    # and continue
    global label_image_content
    image = model.generate_image(
        sv_prompt.get(),
        temperature=temperature,
        top_k=topk,
        supercondition_factor=supercond,
        is_verbose=True
    )
    image.save("out.png")
    label_image_content = PIL.ImageTk.PhotoImage(image)
    label_image.configure(image=label_image_content)

frm = ttk.Frame(root, padding=16)
frm.grid()

props = ttk.Frame(frm)

# outer structure (hbox)
label_image = ttk.Label(frm, image=blank_image)
label_image.grid(column=0, row=0)
ttk.Label(frm, image=padding_image).grid(column=1, row=0)
props.grid(column=2, row=0)

# inner structure (properties and shit)
# prompt field
ttk.Label(props, text="Prompt:").grid(column=0, row=0)
ttk.Entry(props, textvariable=sv_prompt).grid(column=1, row=0)
#
ttk.Label(props, image=padding_image).grid(column=0, row=1)
# temperature field
ttk.Label(props, text="Temperature:").grid(column=0, row=2)
ttk.Entry(props, textvariable=sv_temperature).grid(column=1, row=2)
#
ttk.Label(props, image=padding_image).grid(column=0, row=3)
# topk field
ttk.Label(props, text="Top-K:").grid(column=0, row=4)
ttk.Entry(props, textvariable=sv_topk).grid(column=1, row=4)
#
ttk.Label(props, image=padding_image).grid(column=0, row=5)
# superconditioning field
ttk.Label(props, text="Supercondition Factor:").grid(column=0, row=6)
ttk.Entry(props, textvariable=sv_supercond).grid(column=1, row=6)
#
ttk.Label(props, image=padding_image).grid(column=0, row=7)
# buttons
ttk.Button(props, text="Generate", command=generate).grid(column=0, row=8)
ttk.Button(props, text="Quit", command=root.destroy).grid(column=1, row=8)

# alrighty
root.mainloop()


from min_dalle import MinDalle
import sys
import PIL
import PIL.Image
import PIL.ImageTk
import tkinter
from tkinter import ttk

def regen_root():
    global root
    global blank_image
    global padding_image

    root = tkinter.Tk()
    root.wm_resizable(False, False)

    blank_image = PIL.ImageTk.PhotoImage(PIL.Image.new(size=(256 * 2, 256 * 2), mode="RGB"))
    padding_image = PIL.ImageTk.PhotoImage(PIL.Image.new(size=(16, 16), mode="RGBA"))

regen_root()

is_mega = None
def set_mega_true_and_destroy():
	global is_mega
	is_mega = True
	root.destroy()
def set_mega_false_and_destroy():
	global is_mega
	is_mega = False
	root.destroy()

frm = ttk.Frame(root, padding=16)
frm.grid()
ttk.Button(frm, text="Mega", command=set_mega_true_and_destroy).grid(column=0, row=0)
ttk.Label(frm, image=padding_image).grid(column=1, row=0)
ttk.Button(frm, text="Mini", command=set_mega_false_and_destroy).grid(column=2, row=0)
root.mainloop()

if is_mega is None:
	print("no option selected")
	sys.exit(0)

print("is_mega", is_mega)

model = MinDalle(
    models_root="./pretrained",
    is_mega=is_mega, 
    is_reusable=True,
    is_verbose=True
)

regen_root()

label_image_content = blank_image

sv_prompt = tkinter.StringVar(value="artificial intelligence")
sv_temperature = tkinter.StringVar(value="1")
sv_topk = tkinter.StringVar(value="128")
sv_supercond = tkinter.StringVar(value="16")
bv_seamless = tkinter.BooleanVar(value=False)

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
    try:
        is_seamless = bool(bv_seamless.get())
    except:
        return
    # and continue
    global label_image_content
    image_stream = model.generate_image_stream(
        sv_prompt.get(),
        grid_size=2,
        seed=-1,
        progressive_outputs=True,
        is_seamless=is_seamless,
        temperature=temperature,
        top_k=topk,
        supercondition_factor=supercond,
        is_verbose=True
    )
    for image in image_stream:
        global final_image
        final_image = image
        label_image_content = PIL.ImageTk.PhotoImage(image)
        label_image.configure(image=label_image_content)
        label_image.update()

def save():
    final_image.save('generated/out.png')

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
# seamless
ttk.Label(props, text="Seamless:").grid(column=0, row=8)
ttk.Checkbutton(props, variable=bv_seamless).grid(column=1, row=8)
#
ttk.Label(props, image=padding_image).grid(column=0, row=9)
# buttons
ttk.Button(props, text="Generate", command=generate).grid(column=0, row=10)
ttk.Button(props, text="Quit", command=root.destroy).grid(column=1, row=10)
ttk.Button(props, text="Save", command=save).grid(column=2, row=10)

root.mainloop()
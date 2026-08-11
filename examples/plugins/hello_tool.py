from flimkit.plugins import tool

FLIMKIT_PLUGIN_API = 1


@tool(id='hello_example', label='Hello Plugin...', menu='Tools', order=900)
def open_hello(app):
    from tkinter import messagebox
    messagebox.showinfo('Hello', 'This window came from an add-on, not from FLIMKit.')

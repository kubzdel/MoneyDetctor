import skimage.data as data
img_templates = ((.05, "5gr.png"), (1, "1zl.png"))

templates = []
'''
Returns templates in grayscale
'''
def load_templates():
    import os
    for img in img_templates:
        templates.append(data.load(os.getcwd() + "//templates//" + img[1]), True)
    return templates
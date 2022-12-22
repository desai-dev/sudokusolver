# from PIL import Image, ImageFilter
# import random

# def augment_data():
#     for i in range(253):
#         random_blur = random.randint(0, 1)
#         random_x_transform = random.uniform(-4, 4)
#         random_y_transform = random.uniform(-4, 4)

#         img = Image.open('C:/Users/Dev/Desktop/Sudoku Solver/New Database/9/num9_3.png')
#         img = img.transform(img.size, Image.AFFINE, (1, 0, random_x_transform, 0, 1, random_y_transform))

#         if random_blur == 0:
#             img = img.filter(ImageFilter.BLUR)

#         img.save(f"C:/Users/Dev/Desktop/Sudoku Solver/New Database/9/num9_{i + 763}.png")

# augment_data()
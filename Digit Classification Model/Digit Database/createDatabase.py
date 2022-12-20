# from PIL import Image, ImageDraw, ImageFont
# import random

# def create_database():
#     for i in range(1, 10): 
#         for j in range(1017):
#             random_bg = random.randint(225, 240)
#             random_x_pos = random.uniform(6.5, 9.5)
#             random_y_pos = random.uniform(4, 5)
#             random_num_colour = random.randint(45, 85) 
#             random_font_size = random.randint(20, 21)
#             random_font = random.randint(0, 1)

#             fonts = ["arial.ttf", "georgia.ttf"]

#             fnt = ImageFont.truetype(f"/Fonts/{fonts[random_font]}", random_font_size)

#             img = Image.new("L", (28, 28), random_bg) 
#             img_new = ImageDraw.Draw(img)
#             img_new.text((random_x_pos, random_y_pos), f"{i}", font=fnt, fill=random_num_colour) 
#             img.save(f"C:/Users/Dev/Desktop/Sudoku Solver/Digit Classification Model/Digit Database/{i}/num{i}_{j}.png")

# create_database()
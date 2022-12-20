from PIL import Image, ImageDraw, ImageFont
import random

# make a function, create_database

for i in range(2): # change range to 1000
    random_bg = random.randint(167, 187)
    random_x_pos = random.uniform(8, 12.5) 
    random_y_pos = random.uniform(8, 12.5) 
    random_num_colour = random.randint(45, 85) 
    fnt = ImageFont.truetype("/Fonts/arial.ttf")

    img = Image.new("L", (28, 28), random_bg) 
    img_new = ImageDraw.Draw(img)
    img_new.text((random_x_pos, random_y_pos), "1", font=fnt, fill=random_num_colour) 

    img.save(f"C:/Users/Dev/Desktop/Sudoku Solver/Digit Classification Model/Digit Database/1/NUMBER{i}.png")
from PIL import Image, ImageDraw

# Open the image
image = Image.open("1.jpg")

# Calculate the dimensions of the gradient line
gradient_width = int(image.width * 3/10)
gradient_height = 10

# Create a new image for the gradient line
gradient_line = Image.new("RGB", (gradient_width, gradient_height), color=0)

# Draw the gradient line
draw = ImageDraw.Draw(gradient_line)
for x in range(gradient_width):
    color = (int(255 * (x / gradient_width)), int(255 * (1 - x / gradient_width)), int(255 * (x / gradient_width)))
    draw.line([(x, 0), (x, gradient_height)], fill=color, width=1)

# Get the value from the user and calculate the position of the pointer
pointer_value = int(input("Enter the value for the pointer (between 0 and 255): "))
pointer_position = int(pointer_value / 255 * gradient_width)

# Draw the pointer on the gradient line
draw.rectangle([(pointer_position-2, 0), (pointer_position+2, gradient_height)], fill=(255, 255, 255))

# Paste the gradient line onto the original image
x_pos = image.width - gradient_width
y_pos = 0
image.paste(gradient_line, box=(x_pos, y_pos))

# Save the modified image
image.save("image_with_gradient_line.jpg")

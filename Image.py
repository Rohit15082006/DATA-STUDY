from PIL import Image, ImageDraw, ImageFont

# Create a new white background image
image_width = 800
image_height = 200
image = Image.new('RGB', (image_width, image_height), 'white')
draw = ImageDraw.Draw(image)

# Load fonts (adjust font path if needed)
try:
    title_font = ImageFont.truetype("arial.ttf", 40)
    subtitle_font = ImageFont.truetype("arial.ttf", 30)
except IOError:
    # Fallback to default font if Arial not available
    title_font = ImageFont.load_default()
    subtitle_font = ImageFont.load_default()

# Text content
text_lines = [
    ("Digital Art Project", title_font),
    ("Made by Rohit", subtitle_font)  # Fixed name typo from "Ront" to "Rohit"
]

# Calculate vertical positioning
total_text_height = sum(font.getbbox(text)[3] - font.getbbox(text)[1] for text, font in text_lines)
y_position = (image_height - total_text_height) // 2

# Draw each line centered horizontally
for text, font in text_lines:
    bbox = font.getbbox(text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x_position = (image_width - text_width) // 2
    draw.text((x_position, y_position), text, font=font, fill='black')
    y_position += text_height + 10  # Add spacing between lines

# Save the image
image.save('digital_art_project.png')
print("Image created successfully!")
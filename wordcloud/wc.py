from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

d = path.dirname(__file__)

# Read the whole text.
train_data = pd.read_csv('../datasets/train_set.csv', sep="\t")

coloring = np.array(Image.open(path.join(d, "colors-6.jpg")))
stopwords = set(STOPWORDS)
stopwords.add("said")
stopwords.add("say")
politics = ""
film = ""
football = ""
business = ""
technology = ""

for index, row in train_data.iterrows():
	if row['Category'] == "Politics":
		politics += row["Content"] + " "
	elif row['Category'] == "Film":
		film += row["Content"] + " "
	elif row["Category"] == "Football":
		football += row["Content"] + " "
	elif row["Category"] == "Business":
		business += row["Content"] + " "
	elif row["Category"] == "Technology":
		technology += row["Content"] + " "
	# film += row["Fiml"]
	# football += row["Football"]
	# business += row["Business"]
	# technology += row["Technology"]

# create coloring from image
image_colors = ImageColorGenerator(coloring)

# recolor wordcloud and show
# we could also give color_func=image_colors directly in the constructor
wc_politics = WordCloud(background_color="white", max_words=2000, mask=coloring,
               stopwords=stopwords, max_font_size=200, random_state=42, width = 1920, height = 1080).generate(politics)
plt.imshow(wc_politics.recolor(color_func=image_colors), interpolation="bilinear")
wc_politics.to_file("./politics_wc.png")

wc_film = WordCloud(background_color="white", max_words=2000, mask=coloring,
               stopwords=stopwords, max_font_size=200, random_state=42, width = 1920, height = 1080).generate(film)
plt.imshow(wc_film.recolor(color_func=image_colors), interpolation="bilinear")
wc_film.to_file("./film_wc.png")

wc_football = WordCloud(background_color="white", max_words=2000, mask=coloring,
               stopwords=stopwords, max_font_size=200, random_state=42, width = 1920, height = 1080).generate(football)
plt.imshow(wc_football.recolor(color_func=image_colors), interpolation="bilinear")
wc_football.to_file("./football_wc.png")

wc_business =WordCloud(background_color="white", max_words=2000, mask=coloring,
               stopwords=stopwords, max_font_size=200, random_state=42, width = 1920, height = 1080).generate(business)
plt.imshow(wc_business.recolor(color_func=image_colors), interpolation="bilinear")
wc_business.to_file("./business_wc.png")

wc_technology = WordCloud(background_color="white", max_words=2000, mask=coloring,
               stopwords=stopwords, max_font_size=200, random_state=42, width = 1920, height = 1080).generate(technology)
plt.imshow(wc_technology.recolor(color_func=image_colors), interpolation="bilinear")
wc_technology.to_file("./technology_wc.png")

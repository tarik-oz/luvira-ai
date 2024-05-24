import os

# Dosyaların bulunduğu dizin
image_dir = "data/images"

# Dosya adı olarak yazılacak txt dosyası
txt_file = "image_names.txt"

# Dosya adlarını al
image_names = os.listdir(image_dir)

# Sayı büyüklüğüne göre sırala
image_names_sorted = sorted(image_names, key=lambda x: int(x.split(".")[0]))

# Dosyayı yazma modunda aç
with open(txt_file, "w") as f:
    # Sıralanmış dosya adlarını yaz
    for filename in image_names_sorted:
        if filename.endswith(".jpg"):
            f.write(filename[:-4] + "\n")

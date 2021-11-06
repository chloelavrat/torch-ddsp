import youtube_dl

# -- Nobuya Sugawa plays J.S.BACH Saxophone Solo Recital
youtube_url = "https://www.youtube.com/watch?v=dJvkQxe-z4Q"
mix_filename = "audio_database.wav" 
# -- Download and format informations
ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': mix_filename,
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192',
    }],
    'quiet': True,
    'restrictfilenames': True}

# -- Download
ydl = youtube_dl.YoutubeDL(ydl_opts)
ydl.download([youtube_url])
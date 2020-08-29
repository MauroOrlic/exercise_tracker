from pytube import YouTube

yt = YouTube('https://www.youtube.com/watch?v=jfLZygw-L0o')
yt.streams.first()

yt.streams[0].download()

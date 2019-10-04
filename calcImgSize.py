imgX = int(input('Enter image size - x: '))
imgY = int(input('Enter image size - y: '))
imgZ = int(input('Enter image size - z: '))

pxls = imgX * imgY * imgZ

bitTot = pxls*32

byteTot = bitTot/8      # bytes
byteTot = byteTot/1024  # kb
byteTot = byteTot/1024  # mb
byteTot = byteTot/1024  # gb

print('Memory taken in GB = ',byteTot)
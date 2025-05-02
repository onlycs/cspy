from JES import *
from os import listdir
from math import ceil

files = listdir("assets/music")
music = [makeSound("assets/music/" + file) for file in files]
output = [makeEmptySound(getLength(music[0])) for _ in range(5)]

# segment 1, mangle trump singing havana by reversing it
havana = music[0]
havanaOut = output[0]
havanaSamples = getSamples(havana)
for i in range(0, getLength(havana)):
    setSampleValueAt(
        havanaOut,
        i,
        getSampleValueAt(havana, getLength(havana) - i - 1)
    )

# segment 2, mangle trump saying china by changing the volume 5 times throughout the piece
china = music[1]
chinaOut = output[1]
chinaSamples = getSamples(china)
segmentLength = getLength(china) / 5
amplify = 1
for i in range(0, getLength(china)):
    setSampleValueAt(
        chinaOut,
        i,
        getSampleValueAt(china, i) * amplify
    )

    if i % segmentLength == 0:
        amplify += 1

# segment 3, blend the two segments together (same length)
blend = output[2]
blendSamples = getSamples(blend)
for i in range(0, getLength(blend)):
    setSampleValueAt(
        blend,
        i,
        (getSampleValueAt(havanaOut, i) + getSampleValueAt(chinaOut, i)) / 2
    )

# segment 4, gradually reduce the volume of the blend
fade = output[3]
fadeSamples = getSamples(fade)
for i in range(0, getLength(fade)):
    setSampleValueAt(
        fade,
        i,
        getSampleValueAt(blend, i) * (1 - i / getLength(fade))
    )

# segment 5, bass-boost havana
bassBoost = output[4]
bassBoostSamples = getSamples(bassBoost)
for i in range(0, getLength(bassBoost)):
    value = getSampleValueAt(havana, i)

    if value <= 0:
        setSampleValueAt(bassBoost, i, value*2)
    else:
        setSampleValueAt(bassBoost, i, -3.2e4)

# string them together with 0.3s silence in between
output = [music[0], music[1], havanaOut, chinaOut, blend, fade, bassBoost]
final = makeEmptySound(int(ceil(
    sum([getLength(sound) for sound in output]) + 0.3 * len(output) * 22050
)))
finalSamples = getSamples(final)
cursor = 0
for sound in output:
    for i in range(0, getLength(sound)):
        setSampleValueAt(final, cursor + i, getSampleValueAt(sound, i))
    cursor += getLength(sound) + int(0.3 * 22050)

# write
writeSoundTo(final, "output.wav")


"""
For evaluation with STRONG data, we will build a labeled test dataset.
This will involve using a file that has the mic'ed audio correctly separated from the replay audio.
This will be used with our simpler voice detection algorithm to generate labels.
We will then create the test audio by adding the mic audio to the game replay audio.
"""

# List the STRONG data files with separated mic and replay audio
# Teams 40-62 (skip 53)
teams = list(range(40, 63))
teams.remove(53)

# initialize the output folder

for team in teams:
    # identify the mic audio

    # identify the video audio

    # identify the synchronization offset

    # generate labels
    # check if mic audio voice detection exists
    # if not, run voice audio detection

    # output labels to output folder

    # merge the video and the mic audio

    # output merged audio to output folder

    pass


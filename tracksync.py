"""Generates pairwise video comparisons between drivers.

This script takes in a csv containing timestamps and speeds for
various drivers' Catalyst videos. Then, it uses one video as a
reference and modulates the speed of the other to create a
comparison video where the cars move at approximately the same
speed.
"""

import csv
import os
import sys

from collections import namedtuple

Segment = namedtuple('Segment', ['title', 'timestamp', 'speed'])

class Video:
    def __init__(self, driver):
        self.driver = driver
        self.segments = []

    def AddSegment(self, segment):
        self.segments.append(segment)

    def GenerateSpeedFilter(self, reference):
        assert len(self.segments) == len(reference.segments)
        scales = []
        for i in range(1, len(self.segments)):
            my_duration = (self.segments[i].timestamp -
                           self.segments[i-1].timestamp)
            ref_duration = (reference.segments[i].timestamp -
                            reference.segments[i-1].timestamp)

            # Ratio > 1.0 if I am slower than the reference.
            # Ratio < 1.0 if I am faster than the reference.
            #
            # Example: I require 10 seconds to do what the
            # reference does in 15 seconds. The scaling ratio
            # for my video should be 1.5 to make them the same
            # length.
            ratio = ref_duration / my_duration
            scales.append(ratio)

        vf = ''
        af = ''
        # We trim all the video preceding the first timestamp by
        # starting with the second segment. We only concatenate
        # between the first segment and final timestamp.
        for i in range(1, len(self.segments)):
            t_prev = self.segments[i-1].timestamp
            t = self.segments[i].timestamp
            r = scales[i-1]

            # Only apply this ratio to the given segment.
            vf += f'[0:v]trim=start={t_prev:.4f}:end={t:.4f},'

            # Apply the ratio to the whole segment.
            vf += f'setpts={r:.4f}*(PTS-STARTPTS)[v{i}];\n'

        f = vf + af
        
        # Now concatenate all the segments to the label [v1out].
        concat = ''.join([f'[v{i}]' for i in range(1, len(self.segments))])
        f += concat + f'concat=n={len(self.segments)-1}:v=1:a=0[v1out];\n'

        # Add the full reference clip to create [v2out].
        r0 = reference.segments[0].timestamp
        rn = reference.segments[-1].timestamp
        f += f'[1:v]trim={r0:.4f}:{rn:.4f}[v2out];'
        f += f'[1:a]atrim={r0:.4f}:{rn:.4f}[a2out];'

        # Now vertically stack the processed videos.
        f += f'[v1out][v2out]vstack'
        
        return f

    
def read_csv(csv_filename):
    videos = []
    with open(csv_filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        num_drivers = 0
        for row in reader:
            # The first row includes the names of the drivers in
            # even rows. The speeds do not have titles.
            if len(videos) == 0:
                drivers = [str(row[i]) for i in range(len(row))
                           if i % 2 == 1]
                num_drivers = len(drivers)
                for d in drivers:
                    videos.append(Video(d))
                continue

            # Each row following the header provides driver segments
            segment_title = str(row[0])

            for i in range(num_drivers):
                segment = Segment(segment_title, float(row[2*i + 1]),
                                  float(row[2*i + 2]))
                videos[i].AddSegment(segment)
    return videos


def generate(video0, video1):
    ftr = video0.GenerateSpeedFilter(video1)

    d0 = video0.driver
    d1 = video1.driver
    output_filename = f'{d0}_v_{d1}.mp4'
    cmd = (f'ffmpeg -y -i ./{d0}.mp4 -i ./{d1}.mp4 ' +
           f'-filter_complex "{ftr}" -map "[a2out]:a" ./{output_filename}')
    os.system(cmd)
    

def main(argv):
    csv_filename = argv[1]
    videos = read_csv(csv_filename)
    
    for v0 in videos:
        for v1 in videos:
            if v0.driver != v1.driver:
                generate(v0, v1)


if __name__ == "__main__":
    main(sys.argv)

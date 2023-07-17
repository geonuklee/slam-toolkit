import glob
from os import path as osp
import cv2

import subprocess

def copy_to_clipboard(text):
    proc = subprocess.Popen(['xclip', '-selection', 'clipboard'], stdin=subprocess.PIPE)
    proc.communicate(input=text.encode('utf-8'))


if __name__=='__main__':
    seqs = glob.glob('output/sequences/*')
    stop = True
    for seq_dir in seqs:
        seq = osp.basename(seq_dir)
        image_files = glob.glob(osp.join(seq_dir,'image_0','*.png') )
        image_files = sorted(image_files, key=lambda x: x.split('/')[-1])
        copy_to_clipboard(seq)
        c = None
        for i, fn in enumerate(image_files):

            rgb = cv2.imread(fn,cv2.IMREAD_COLOR)
            rgb[:25,:,:] = 255
            cv2.putText(rgb, '%s, %d/%d' %(seq, i+1, len(image_files)), (5,15), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0,0,0), 1)
            cv2.imshow("viewer (p)ause (n)ext (q)uit", rgb)
            if stop:
                c = cv2.waitKey()
            else:
                c = cv2.waitKey(1)

            if c==ord('q'):
                exit(1)
            elif c==ord('p'):
                stop = not stop
            elif c==ord('n'):
                break
        if c != ord('n'):
            c = cv2.waitKey()

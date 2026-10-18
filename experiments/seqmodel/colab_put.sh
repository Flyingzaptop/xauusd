#!/bin/bash
# upload large files to a Colab session in 40 MB chunks (the contents API rejects big single uploads), reassemble and md5-check.
# usage: colab_put.sh SESSION REMOTE_DIR FILE [FILE ...]
S=$1;R=$2;shift 2;C=~/.local/bin/colab;T=$(mktemp -d)
for f in "$@"; do
  b=$(basename $f);rm -f $T/*;split -b 40M -d -a 3 $f $T/$b.;n=0
  for c in $(ls $T | sort); do
    for try in 1 2 3 4; do timeout 300 $C upload -s $S $T/$c $R/.parts/$c >/dev/null 2>&1 && break; sleep $((2**try)); done; n=$((n+1))
  done
  m=$(md5sum $f|cut -d' ' -f1)
  printf 'import glob,os,hashlib\np=sorted(glob.glob("%s/.parts/%s.*"));o="%s/%s"\nwith open(o,"wb") as w:\n    for x in p:w.write(open(x,"rb").read());os.remove(x)\nh=hashlib.md5(open(o,"rb").read()).hexdigest();print("OK" if h=="%s" else "BAD",o,len(p))\n' $R $b $R $b $m | timeout 300 $C exec -s $S 2>&1 | tail -1
done
rm -rf $T

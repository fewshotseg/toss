import os

def has_mask(f):
	return os.path.isfile(os.path.join("data/fss1k/masks", f+".png"))

def has_img(f):
	return os.path.isfile(os.path.join("data/fss1k/images", f+".jpg"))


bm = True

with open("data/tiers/general/general_tier.txt") as fd:
	for line in fd:
		q, s = line.strip().split(" ")
		q = "_".join(q.split("_")[1:])
		s = "_".join(s.split("_")[1:])
		if not(has_img(q)) or not(has_mask(q)):
			print("missing", q)
			bm = False
		if not(has_img(s)) or not(has_mask(s)):
			print("missing", s)
			bm = False

if bm:
	print("all verified")
else:
	print("missing files")




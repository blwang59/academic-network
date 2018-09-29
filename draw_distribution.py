import json
author_name = json.load(open('./data/name_per_author.json'))
print(len(author_name))
paper_per_author = json.load(open('data/paper_per_author.json'))

citations = json.load(open('data/citations_per_author.json'))
print(len(citations))
hindex = json.load(open('data/hindex.json'))
print(len(hindex))
import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
# np.random.seed(19680801)
#
# mu, sigma = 100, 15
# x = mu + sigma * np.random.randn(10000)

# the histogram of the data
# n, bins, patches = \
plt.hist(list(paper_per_author.values()), 50, facecolor='g', alpha=0.75)
plt.yscale('log')
plt.xlabel('number of papers')
plt.ylabel('number of people')
plt.title('paper distribution in Microsoft Acadamic')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.savefig('tests/paper_distribution.pdf')
plt.show()

plt.hist(list(citations.values()), 50, facecolor='g', alpha=0.75)
plt.yscale('log')
plt.xlabel('citations')
plt.ylabel('number of people')
plt.title('citations distribution in Microsoft Acadamic')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.savefig('tests/citation_distribution.pdf')
plt.show()

plt.hist(list(hindex.values()), 50, facecolor='g', alpha=0.75)
plt.yscale('log')
plt.xlabel('hindex')
plt.ylabel('number of people')
plt.title('hindex distribution in Microsoft Acadamic')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.savefig('tests/hindex_distribution.pdf')
plt.show()
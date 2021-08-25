import matplotlib.colors as c
import matplotlib.pyplot as plt
import umap


class Umap():
    def __init__(self, data=[], n_class=10, min_dist=0.3):
        self.data = data
        self.n_class = n_class
        self.min_dist = min_dist
        
        self.embedding = umap.UMAP(n_neighbors=n_class,
                      metric='correlation').fit_transform(data)
        
    def umap_plt(self, y_train, name='umap'):
        cmap = plt.get_cmap('rainbow', 10)
        fig = plt.figure()
        ax = plt.axes()
        scatter = ax.scatter(self.embedding[:,0], self.embedding[:,1], 
                              c=y_train, s=20, edgecolor='black', cmap=cmap)

        legend1 = ax.legend(*scatter.legend_elements(), loc="upper right",
                            bbox_to_anchor=(1.2, 1), title="Classes", borderaxespad=0.0)
        #for i, cl in enumerate(classes):
        #    legend1.get_texts()[i].set_text(cl)

        ax.add_artist(legend1)
        #plt.colorbar(drawedges=True)
        plt.savefig('./TSNE_images/tsne_epoch_{}.png'.format(self.name),
                    bbox_inches='tight', dpi=200)
        
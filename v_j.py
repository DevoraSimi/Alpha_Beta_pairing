

def v_j_format(gene, levels, gene_name):
    if gene in ["~", "nan", "", "NA"]:
        return "~"

    if gene_name == "TRBV":
        if "TCRBV" in gene:
            gene = gene.replace("TCRBV", "TRBV")
        if "TRBV" not in gene:
            return "~"
    if gene_name == "TRBJ":
        if not gene.startswith("TRBJ"):
            return "~"
    if "TRDAV" in gene:
        gene = gene.replace("TRDAV", "TRAV")
    if "TRA21" in gene:
        gene = gene.replace("TRA21", "TRAV21")
    if "TRAJF" in gene:
        return "~"
    if "TRDJ" in gene:
        return "~"

    gene = gene.split("/")[0]
    gene_list = gene.replace(" ", "").replace("*", "-").replace(":", "-").split("-")

    #add zero to numbers with one digit
    gene_value = gene_list[0].replace(gene_name, "")
    if len(gene_value) == 1:
        gene_value = "0" + gene_value
    gene_list[0] = gene_name + gene_value

    if levels == 1:
        return gene_list[0]
    if levels == 2:
        if len(gene_list) == 1:
            return gene_list[0]
        return gene_list[0] + "-" + str(int(gene_list[1]))



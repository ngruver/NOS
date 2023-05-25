import transformers

vocab_file = "/home/nvg7279/src/seq-struct/vocab.txt"
tokenizer = transformers.BertTokenizerFast(
    vocab_file=vocab_file, 
    do_lower_case=False,
)

seq = "[AbHC]QVQLQQW-GAGLLKPSETLSLTCAVYG-GSFSG-----YYWSWIRQPPGKGLEWIGEINH----SGSTNYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCASSTIFG-------------------VVGGDYWGQGTLVTVSS[AbLC]QLVLTQS-PSASASLGASVKLTCTLS--SGHSS-----YAIAWHQQQPEKGPRYLMKLNS----DGSHSKGDGIPDRFSGSSSG--AERYLTISSLQSEDEADYYCQTWG--------------------------TEFGGGTKLTVL[Ag]"

replace_dict = {
    "[AbHC]": "1",
    "[AbLC]": "2",
    "[Ag]": "3",
}
for k, v in replace_dict.items():
    seq = seq.replace(k, v)
seq = " ".join(seq)
for k, v in replace_dict.items():
    seq = seq.replace(v, k)

print(seq)
print(tokenizer(seq))
print(tokenizer.special_tokens_map)
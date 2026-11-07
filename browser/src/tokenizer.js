// CLIP byte-level BPE. Model vocabulary/merges are pinned with the export.
export class Tokenizer {
  constructor(vocab,merges) {
    this.vocab=vocab;this.ranks=new Map(merges.trim().split('\n').slice(1).map((x,i)=>[x.trim(),i]));this.cache=new Map();
    const bytes=[...Array(94)].map((_,i)=>i+33).concat([...Array(12)].map((_,i)=>i+161),[...Array(82)].map((_,i)=>i+174));
    const chars=[...bytes];let n=0;
    for(let b=0;b<256;b++)if(!bytes.includes(b)){bytes.push(b);chars.push(256+n++);}
    this.byteMap=new Map(bytes.map((b,i)=>[b,String.fromCodePoint(chars[i])]));
  }
  bpe(token) {
    if(this.cache.has(token))return this.cache.get(token);
    let word=[...token];word[word.length-1]+='</w>';
    while(word.length>1){
      let best=Infinity,pair=null;
      for(let i=0;i<word.length-1;i++){const key=word[i]+' '+word[i+1],rank=this.ranks.get(key);if(rank<best){best=rank;pair=[word[i],word[i+1]];}}
      if(!pair)break;
      const next=[];for(let i=0;i<word.length;i++){if(word[i]===pair[0]&&word[i+1]===pair[1]){next.push(word[i]+word[++i]);}else next.push(word[i]);}word=next;
    }
    this.cache.set(token,word);return word;
  }
  encode(text) {
    const clean=text.normalize('NFC').trim().replace(/\s+/gu,' ').toLowerCase();
    const tokens=clean.match(/<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+/gu)||[];
    const ids=[49406];
    for(const token of tokens){const encoded=[...new TextEncoder().encode(token)].map(b=>this.byteMap.get(b)).join('');for(const piece of this.bpe(encoded)){const id=this.vocab[piece];if(id===undefined)throw Error('Unrecognized prompt token.');ids.push(id);}}
    if(ids.length>76)throw Error('This prompt exceeds the 75-token content limit. Shorten it and retry.');
    ids.push(49407);while(ids.length<77)ids.push(49407);return new Int32Array(ids);
  }
}

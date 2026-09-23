// Original scene 2 from the saved 8-visual gallery, preserved for comparison.
export function galleryNebula(g,w,h,t,freq,gain=1.5) {
 const R=Math.min(w,h)*.3;
 let sum=0;for(let i=1;i<160;i++)sum+=freq[i]||0;
 const e=sum/159/255*gain;
 const sample=i=>Math.min(1,Math.pow((freq[i]||0)/255,2)*gain);
 const dot=(x,y,r,color)=>{g.beginPath();g.arc(x,y,Math.max(.3,r),0,Math.PI*2);g.fillStyle=color;g.fill();};
g.globalCompositeOperation='lighter';for(let k=0;k<950;k++){const a=k*2.399+t*(.08+e*.12),u=((k*73)%947)/947,amp=sample(k%180),r=R*Math.sqrt(u)*(1+amp*.9),z=Math.sin(k*1.7+t*.3);dot(Math.cos(a+r*.013)*r*1.5,Math.sin(a)*r*.65+z*R*.18,(.4+amp*2.5)*(w/750),`hsla(${180+u*110},90%,${55+amp*35}%,${.3+amp*.7})`);}
}

!function(e,r){"function"==typeof define&&define.amd?define(r):"object"==typeof exports?module.exports=r():r()(e.lunr)}(this,function(){return function(e){if(void 0===e)throw new Error("Lunr is not present. Please include / require Lunr before this script.");if(void 0===e.stemmerSupport)throw new Error("Lunr stemmer support is not present. Please include / require Lunr stemmer support before this script.");e.fr=function(){this.pipeline.reset(),this.pipeline.add(e.fr.trimmer,e.fr.stopWordFilter,e.fr.stemmer),this.searchPipeline&&(this.searchPipeline.reset(),this.searchPipeline.add(e.fr.stemmer))},e.fr.wordCharacters="A-Za-zªºÀ-ÖØ-öø-ʸˠ-ˤᴀ-ᴥᴬ-ᵜᵢ-ᵥᵫ-ᵷᵹ-ᶾḀ-ỿⁱⁿₐ-ₜKÅℲⅎⅠ-ↈⱠ-ⱿꜢ-ꞇꞋ-ꞭꞰ-ꞷꟷ-ꟿꬰ-ꭚꭜ-ꭤﬀ-ﬆＡ-Ｚａ-ｚ",e.fr.trimmer=e.trimmerSupport.generateTrimmer(e.fr.wordCharacters),e.Pipeline.registerFunction(e.fr.trimmer,"trimmer-fr"),e.fr.stemmer=function(){var r=e.stemmerSupport.Among,s=e.stemmerSupport.SnowballProgram,i=new function(){function e(e,r,s){return!(!z.eq_s(1,e)||(z.ket=z.cursor,!z.in_grouping(v,97,251)))&&(z.slice_from(r),z.cursor=s,!0)}function i(e,r,s){return!!z.eq_s(1,e)&&(z.ket=z.cursor,z.slice_from(r),z.cursor=s,!0)}function n(){for(;!z.in_grouping(v,97,251);){if(z.cursor>=z.limit)return!0;z.cursor++}for(;!z.out_grouping(v,97,251);){if(z.cursor>=z.limit)return!0;z.cursor++}return!1}function t(){return w<=z.cursor}function u(){return l<=z.cursor}function o(){return a<=z.cursor}function c(){if(!function(){var e,r;if(z.ket=z.cursor,e=z.find_among_b(d,43)){switch(z.bra=z.cursor,e){case 1:if(!o())return!1;z.slice_del();break;case 2:if(!o())return!1;z.slice_del(),z.ket=z.cursor,z.eq_s_b(2,"ic")&&(z.bra=z.cursor,o()?z.slice_del():z.slice_from("iqU"));break;case 3:if(!o())return!1;z.slice_from("log");break;case 4:if(!o())return!1;z.slice_from("u");break;case 5:if(!o())return!1;z.slice_from("ent");break;case 6:if(!t())return!1;if(z.slice_del(),z.ket=z.cursor,e=z.find_among_b(_,6))switch(z.bra=z.cursor,e){case 1:o()&&(z.slice_del(),z.ket=z.cursor,z.eq_s_b(2,"at")&&(z.bra=z.cursor,o()&&z.slice_del()));break;case 2:o()?z.slice_del():u()&&z.slice_from("eux");break;case 3:o()&&z.slice_del();break;case 4:t()&&z.slice_from("i")}break;case 7:if(!o())return!1;if(z.slice_del(),z.ket=z.cursor,e=z.find_among_b(b,3))switch(z.bra=z.cursor,e){case 1:o()?z.slice_del():z.slice_from("abl");break;case 2:o()?z.slice_del():z.slice_from("iqU");break;case 3:o()&&z.slice_del()}break;case 8:if(!o())return!1;if(z.slice_del(),z.ket=z.cursor,z.eq_s_b(2,"at")&&(z.bra=z.cursor,o()&&(z.slice_del(),z.ket=z.cursor,z.eq_s_b(2,"ic")))){z.bra=z.cursor,o()?z.slice_del():z.slice_from("iqU");break}break;case 9:z.slice_from("eau");break;case 10:if(!u())return!1;z.slice_from("al");break;case 11:if(o())z.slice_del();else{if(!u())return!1;z.slice_from("eux")}break;case 12:if(!u()||!z.out_grouping_b(v,97,251))return!1;z.slice_del();break;case 13:return t()&&z.slice_from("ant"),!1;case 14:return t()&&z.slice_from("ent"),!1;case 15:return r=z.limit-z.cursor,z.in_grouping_b(v,97,251)&&t()&&(z.cursor=z.limit-r,z.slice_del()),!1}return!0}return!1}()&&(z.cursor=z.limit,!function(){var e,r;if(z.cursor<w)return!1;if(r=z.limit_backward,z.limit_backward=w,z.ket=z.cursor,!(e=z.find_among_b(k,35)))return z.limit_backward=r,!1;if(z.bra=z.cursor,1==e){if(!z.out_grouping_b(v,97,251))return z.limit_backward=r,!1;z.slice_del()}return z.limit_backward=r,!0}()&&(z.cursor=z.limit,!function(){var e,r,s;if(z.cursor<w)return!1;if(r=z.limit_backward,z.limit_backward=w,z.ket=z.cursor,!(e=z.find_among_b(p,38)))return z.limit_backward=r,!1;switch(z.bra=z.cursor,e){case 1:if(!o())return z.limit_backward=r,!1;z.slice_del();break;case 2:z.slice_del();break;case 3:z.slice_del(),s=z.limit-z.cursor,z.ket=z.cursor,z.eq_s_b(1,"e")?(z.bra=z.cursor,z.slice_del()):z.cursor=z.limit-s}return z.limit_backward=r,!0}())))return z.cursor=z.limit,void function(){var e,r,s,i,n=z.limit-z.cursor;if(z.ket=z.cursor,z.eq_s_b(1,"s")?(z.bra=z.cursor,r=z.limit-z.cursor,z.out_grouping_b(h,97,232)?(z.cursor=z.limit-r,z.slice_del()):z.cursor=z.limit-n):z.cursor=z.limit-n,z.cursor>=w){if(s=z.limit_backward,z.limit_backward=w,z.ket=z.cursor,e=z.find_among_b(g,7))switch(z.bra=z.cursor,e){case 1:if(o()){if(i=z.limit-z.cursor,!z.eq_s_b(1,"s")&&(z.cursor=z.limit-i,!z.eq_s_b(1,"t")))break;z.slice_del()}break;case 2:z.slice_from("i");break;case 3:z.slice_del();break;case 4:z.eq_s_b(2,"gu")&&z.slice_del()}z.limit_backward=s}}();z.cursor=z.limit,z.ket=z.cursor,z.eq_s_b(1,"Y")?(z.bra=z.cursor,z.slice_from("i")):(z.cursor=z.limit,z.eq_s_b(1,"ç")&&(z.bra=z.cursor,z.slice_from("c")))}var a,l,w,f=[new r("col",-1,-1),new r("par",-1,-1),new r("tap",-1,-1)],m=[new r("",-1,4),new r("I",0,1),new r("U",0,2),new r("Y",0,3)],_=[new r("iqU",-1,3),new r("abl",-1,3),new r("Ièr",-1,4),new r("ièr",-1,4),new r("eus",-1,2),new r("iv",-1,1)],b=[new r("ic",-1,2),new r("abil",-1,1),new r("iv",-1,3)],d=[new r("iqUe",-1,1),new r("atrice",-1,2),new r("ance",-1,1),new r("ence",-1,5),new r("logie",-1,3),new r("able",-1,1),new r("isme",-1,1),new r("euse",-1,11),new r("iste",-1,1),new r("ive",-1,8),new r("if",-1,8),new r("usion",-1,4),new r("ation",-1,2),new r("ution",-1,4),new r("ateur",-1,2),new r("iqUes",-1,1),new r("atrices",-1,2),new r("ances",-1,1),new r("ences",-1,5),new r("logies",-1,3),new r("ables",-1,1),new r("ismes",-1,1),new r("euses",-1,11),new r("istes",-1,1),new r("ives",-1,8),new r("ifs",-1,8),new r("usions",-1,4),new r("ations",-1,2),new r("utions",-1,4),new r("ateurs",-1,2),new r("ments",-1,15),new r("ements",30,6),new r("issements",31,12),new r("ités",-1,7),new r("ment",-1,15),new r("ement",34,6),new r("issement",35,12),new r("amment",34,13),new r("emment",34,14),new r("aux",-1,10),new r("eaux",39,9),new r("eux",-1,1),new r("ité",-1,7)],k=[new r("ira",-1,1),new r("ie",-1,1),new r("isse",-1,1),new r("issante",-1,1),new r("i",-1,1),new r("irai",4,1),new r("ir",-1,1),new r("iras",-1,1),new r("ies",-1,1),new r("îmes",-1,1),new r("isses",-1,1),new r("issantes",-1,1),new r("îtes",-1,1),new r("is",-1,1),new r("irais",13,1),new r("issais",13,1),new r("irions",-1,1),new r("issions",-1,1),new r("irons",-1,1),new r("issons",-1,1),new r("issants",-1,1),new r("it",-1,1),new r("irait",21,1),new r("issait",21,1),new r("issant",-1,1),new r("iraIent",-1,1),new r("issaIent",-1,1),new r("irent",-1,1),new r("issent",-1,1),new r("iront",-1,1),new r("ît",-1,1),new r("iriez",-1,1),new r("issiez",-1,1),new r("irez",-1,1),new r("issez",-1,1)],p=[new r("a",-1,3),new r("era",0,2),new r("asse",-1,3),new r("ante",-1,3),new r("ée",-1,2),new r("ai",-1,3),new r("erai",5,2),new r("er",-1,2),new r("as",-1,3),new r("eras",8,2),new r("âmes",-1,3),new r("asses",-1,3),new r("antes",-1,3),new r("âtes",-1,3),new r("ées",-1,2),new r("ais",-1,3),new r("erais",15,2),new r("ions",-1,1),new r("erions",17,2),new r("assions",17,3),new r("erons",-1,2),new r("ants",-1,3),new r("és",-1,2),new r("ait",-1,3),new r("erait",23,2),new r("ant",-1,3),new r("aIent",-1,3),new r("eraIent",26,2),new r("èrent",-1,2),new r("assent",-1,3),new r("eront",-1,2),new r("ât",-1,3),new r("ez",-1,2),new r("iez",32,2),new r("eriez",33,2),new r("assiez",33,3),new r("erez",32,2),new r("é",-1,2)],g=[new r("e",-1,3),new r("Ière",0,2),new r("ière",0,2),new r("ion",-1,1),new r("Ier",-1,2),new r("ier",-1,2),new r("ë",-1,4)],q=[new r("ell",-1,-1),new r("eill",-1,-1),new r("enn",-1,-1),new r("onn",-1,-1),new r("ett",-1,-1)],v=[17,65,16,1,0,0,0,0,0,0,0,0,0,0,0,128,130,103,8,5],h=[1,65,20,0,0,0,0,0,0,0,0,0,0,0,0,0,128],z=new s;this.setCurrent=function(e){z.setCurrent(e)},this.getCurrent=function(){return z.getCurrent()},this.stem=function(){var r=z.cursor;return function(){for(var r,s;;){if(r=z.cursor,z.in_grouping(v,97,251)){if(z.bra=z.cursor,s=z.cursor,e("u","U",r))continue;if(z.cursor=s,e("i","I",r))continue;if(z.cursor=s,i("y","Y",r))continue}if(z.cursor=r,z.bra=r,!e("y","Y",r)){if(z.cursor=r,z.eq_s(1,"q")&&(z.bra=z.cursor,i("u","U",r)))continue;if(z.cursor=r,r>=z.limit)return;z.cursor++}}}(),z.cursor=r,function(){var e=z.cursor;if(w=z.limit,l=w,a=w,z.in_grouping(v,97,251)&&z.in_grouping(v,97,251)&&z.cursor<z.limit)z.cursor++;else if(z.cursor=e,!z.find_among(f,3)){z.cursor=e;do{if(z.cursor>=z.limit){z.cursor=w;break}z.cursor++}while(!z.in_grouping(v,97,251))}w=z.cursor,z.cursor=e,n()||(l=z.cursor,n()||(a=z.cursor))}(),z.limit_backward=r,z.cursor=z.limit,c(),z.cursor=z.limit,function(){var e=z.limit-z.cursor;z.find_among_b(q,5)&&(z.cursor=z.limit-e,z.ket=z.cursor,z.cursor>z.limit_backward&&(z.cursor--,z.bra=z.cursor,z.slice_del()))}(),z.cursor=z.limit,function(){for(var e,r=1;z.out_grouping_b(v,97,251);)r--;if(r<=0){if(z.ket=z.cursor,e=z.limit-z.cursor,!z.eq_s_b(1,"é")&&(z.cursor=z.limit-e,!z.eq_s_b(1,"è")))return;z.bra=z.cursor,z.slice_from("e")}}(),z.cursor=z.limit_backward,function(){for(var e,r;r=z.cursor,z.bra=r,e=z.find_among(m,4);)switch(z.ket=z.cursor,e){case 1:z.slice_from("i");break;case 2:z.slice_from("u");break;case 3:z.slice_from("y");break;case 4:if(z.cursor>=z.limit)return;z.cursor++}}(),!0}};return function(e){return"function"==typeof e.update?e.update(function(e){return i.setCurrent(e),i.stem(),i.getCurrent()}):(i.setCurrent(e),i.stem(),i.getCurrent())}}(),e.Pipeline.registerFunction(e.fr.stemmer,"stemmer-fr"),e.fr.stopWordFilter=e.generateStopWordFilter("ai aie aient aies ait as au aura aurai auraient aurais aurait auras aurez auriez aurions aurons auront aux avaient avais avait avec avez aviez avions avons ayant ayez ayons c ce ceci celà ces cet cette d dans de des du elle en es est et eu eue eues eurent eus eusse eussent eusses eussiez eussions eut eux eûmes eût eûtes furent fus fusse fussent fusses fussiez fussions fut fûmes fût fûtes ici il ils j je l la le les leur leurs lui m ma mais me mes moi mon même n ne nos notre nous on ont ou par pas pour qu que quel quelle quelles quels qui s sa sans se sera serai seraient serais serait seras serez seriez serions serons seront ses soi soient sois soit sommes son sont soyez soyons suis sur t ta te tes toi ton tu un une vos votre vous y à étaient étais était étant étiez étions été étée étées étés êtes".split(" ")),e.Pipeline.registerFunction(e.fr.stopWordFilter,"stopWordFilter-fr")}});
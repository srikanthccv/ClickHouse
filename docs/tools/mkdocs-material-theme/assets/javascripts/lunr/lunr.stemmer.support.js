!function(r,t){"function"==typeof define&&define.amd?define(t):"object"==typeof exports?module.exports=t():t()(r.lunr)}(this,function(){return function(r){r.stemmerSupport={Among:function(r,t,i,s){if(this.toCharArray=function(r){for(var t=r.length,i=new Array(t),s=0;s<t;s++)i[s]=r.charCodeAt(s);return i},!r&&""!=r||!t&&0!=t||!i)throw"Bad Among initialisation: s:"+r+", substring_i: "+t+", result: "+i;this.s_size=r.length,this.s=this.toCharArray(r),this.substring_i=t,this.result=i,this.method=s},SnowballProgram:function(){var r;return{bra:0,ket:0,limit:0,cursor:0,limit_backward:0,setCurrent:function(t){r=t,this.cursor=0,this.limit=t.length,this.limit_backward=0,this.bra=this.cursor,this.ket=this.limit},getCurrent:function(){var t=r;return r=null,t},in_grouping:function(t,i,s){if(this.cursor<this.limit){var e=r.charCodeAt(this.cursor);if(e<=s&&e>=i&&t[(e-=i)>>3]&1<<(7&e))return this.cursor++,!0}return!1},in_grouping_b:function(t,i,s){if(this.cursor>this.limit_backward){var e=r.charCodeAt(this.cursor-1);if(e<=s&&e>=i&&t[(e-=i)>>3]&1<<(7&e))return this.cursor--,!0}return!1},out_grouping:function(t,i,s){if(this.cursor<this.limit){var e=r.charCodeAt(this.cursor);if(e>s||e<i)return this.cursor++,!0;if(!(t[(e-=i)>>3]&1<<(7&e)))return this.cursor++,!0}return!1},out_grouping_b:function(t,i,s){if(this.cursor>this.limit_backward){var e=r.charCodeAt(this.cursor-1);if(e>s||e<i)return this.cursor--,!0;if(!(t[(e-=i)>>3]&1<<(7&e)))return this.cursor--,!0}return!1},eq_s:function(t,i){if(this.limit-this.cursor<t)return!1;for(var s=0;s<t;s++)if(r.charCodeAt(this.cursor+s)!=i.charCodeAt(s))return!1;return this.cursor+=t,!0},eq_s_b:function(t,i){if(this.cursor-this.limit_backward<t)return!1;for(var s=0;s<t;s++)if(r.charCodeAt(this.cursor-t+s)!=i.charCodeAt(s))return!1;return this.cursor-=t,!0},find_among:function(t,i){for(var s=0,e=i,n=this.cursor,u=this.limit,o=0,h=0,c=!1;;){for(var a=s+(e-s>>1),f=0,l=o<h?o:h,_=t[a],m=l;m<_.s_size;m++){if(n+l==u){f=-1;break}if(f=r.charCodeAt(n+l)-_.s[m])break;l++}if(f<0?(e=a,h=l):(s=a,o=l),e-s<=1){if(s>0||e==s||c)break;c=!0}}for(;;){if(o>=(_=t[s]).s_size){if(this.cursor=n+_.s_size,!_.method)return _.result;var b=_.method();if(this.cursor=n+_.s_size,b)return _.result}if((s=_.substring_i)<0)return 0}},find_among_b:function(t,i){for(var s=0,e=i,n=this.cursor,u=this.limit_backward,o=0,h=0,c=!1;;){for(var a=s+(e-s>>1),f=0,l=o<h?o:h,_=(m=t[a]).s_size-1-l;_>=0;_--){if(n-l==u){f=-1;break}if(f=r.charCodeAt(n-1-l)-m.s[_])break;l++}if(f<0?(e=a,h=l):(s=a,o=l),e-s<=1){if(s>0||e==s||c)break;c=!0}}for(;;){var m;if(o>=(m=t[s]).s_size){if(this.cursor=n-m.s_size,!m.method)return m.result;var b=m.method();if(this.cursor=n-m.s_size,b)return m.result}if((s=m.substring_i)<0)return 0}},replace_s:function(t,i,s){var e=s.length-(i-t),n=r.substring(0,t),u=r.substring(i);return r=n+s+u,this.limit+=e,this.cursor>=i?this.cursor+=e:this.cursor>t&&(this.cursor=t),e},slice_check:function(){if(this.bra<0||this.bra>this.ket||this.ket>this.limit||this.limit>r.length)throw"faulty slice operation"},slice_from:function(r){this.slice_check(),this.replace_s(this.bra,this.ket,r)},slice_del:function(){this.slice_from("")},insert:function(r,t,i){var s=this.replace_s(r,t,i);r<=this.bra&&(this.bra+=s),r<=this.ket&&(this.ket+=s)},slice_to:function(){return this.slice_check(),r.substring(this.bra,this.ket)},eq_v_b:function(r){return this.eq_s_b(r.length,r)}}}},r.trimmerSupport={generateTrimmer:function(r){var t=new RegExp("^[^"+r+"]+"),i=new RegExp("[^"+r+"]+$");return function(r){return"function"==typeof r.update?r.update(function(r){return r.replace(t,"").replace(i,"")}):r.replace(t,"").replace(i,"")}}}}});
using System;
using System.IO;
using System.Linq;

public static class SevenStageSavedAudit {
 public static ulong Hash(byte[] bytes) { unchecked { ulong h=14695981039346656037UL; foreach(byte b in bytes){h^=b;h*=1099511628211UL;} return h; } }
 public static byte[] Payload(string path) {
  var all=File.ReadAllBytes(path);
  using(var r=new BinaryReader(new MemoryStream(all))) {
   if(r.ReadUInt64()!=0x31504B434C524443UL) throw new Exception("wrapper magic: "+path);
   var n=r.ReadUInt64(); var h=r.ReadUInt64();
   if(n!=(ulong)all.Length-24) throw new Exception("length: "+path);
   var bytes=r.ReadBytes(checked((int)n)); if(Hash(bytes)!=h) throw new Exception("checksum: "+path); return bytes;
  }
 }
 static float[] Floats(BinaryReader r) {
  var n=r.ReadUInt64(); if(n>100000) throw new Exception("array bound");
  var x=new float[(int)n];
  for(int i=0;i<x.Length;i++){x[i]=r.ReadSingle();if(float.IsNaN(x[i])||float.IsInfinity(x[i]))throw new Exception("nonfinite float");}
  return x;
 }
 public static bool Equal(byte[] a,byte[] b){return a.SequenceEqual(b);}
 public static bool BitsEqual(float a,float b){return Equal(BitConverter.GetBytes(a),BitConverter.GetBytes(b));}
 public static bool FloatEqual(float[] a,float[] b){return a.Length==b.Length&&Enumerable.Range(0,a.Length).All(i=>BitsEqual(a[i],b[i]));}
 public static ulong WeightHash(float[] values){using(var m=new MemoryStream())using(var w=new BinaryWriter(m)){w.Write((ulong)values.Length);foreach(var v in values)w.Write(v);return Hash(m.ToArray());}}
 public sealed class Actor {public uint version,kind,schema,inputs,outputs,study,material;public float[] frozenBase,weights,skip=new float[0];}
 public static Actor ReadActor(string path) {
  using(var r=new BinaryReader(new MemoryStream(Payload(path)))) {
   var a=new Actor(); if(r.ReadUInt32()!=0x4f594654)throw new Exception("actor magic"); a.version=r.ReadUInt32();
   if(a.version==1){a.kind=1;a.inputs=31;a.outputs=1;a.weights=Floats(r);a.frozenBase=a.weights.Take(4).ToArray();}
   else if(a.version==2){a.kind=r.ReadUInt32();a.schema=r.ReadUInt32();a.inputs=r.ReadUInt32();a.outputs=r.ReadUInt32();a.frozenBase=new float[4];for(int i=0;i<4;i++)a.frozenBase[i]=r.ReadSingle();a.weights=Floats(r);if(a.kind!=2||a.schema!=1||a.inputs!=31||a.outputs!=1)throw new Exception("phase actor layout");}
   else if(a.version==3){a.kind=r.ReadUInt32();a.schema=r.ReadUInt32();a.inputs=r.ReadUInt32();a.outputs=r.ReadUInt32();a.study=r.ReadUInt32();a.material=r.ReadUInt32();a.frozenBase=new float[4];for(int i=0;i<4;i++)a.frozenBase[i]=r.ReadSingle();a.skip=Floats(r);a.weights=Floats(r);if(a.kind<3||a.kind>5||a.schema!=2||a.inputs!=31||a.outputs!=(a.kind==3?1u:3u)||a.study<1||a.study>5||a.material>1)throw new Exception("program actor layout");}else throw new Exception("actor version");
   if(r.BaseStream.Position!=r.BaseStream.Length||a.weights.Length!=(a.kind==5?563:a.kind==4?96:32))throw new Exception("actor size/trailing bytes");return a;
  }
 }
 public sealed class Session {
  public string task,parentTask; public int laneSize,countSize,update,actorSteps,criticSteps,parentUpdate,parentActorSteps,parentCriticSteps;public uint version,feedback,kind,feature,practice; public ulong parentPayload,parentInitialWeights,skipHash;public uint study,material;public float targetSpeed,desiredRate;public int samplingOrigin; public byte[] nativeWorldBytes,optimizerBytes;
  public ulong reliable,low;public uint measured,rejected;public int[] config;
  public float[] constants,actor,critic,actorM,actorV,criticM,criticV,actorAdamConfig,criticAdamConfig,initialCoefficients;
  public ulong initialWeights,gait,offset,starts,rng,worlds,counters,attempted,accepted,actions,hash,identityHash;
  public uint startsCount,horizons,clearance,posture,left,right;
 }
 static int Adam(BinaryReader r,out float[] m,out float[] v,out float[] config){config=new float[5];for(int i=0;i<5;i++){config[i]=r.ReadSingle();if(float.IsNaN(config[i])||float.IsInfinity(config[i]))throw new Exception("Adam config");}var age=r.ReadInt32();m=Floats(r);v=Floats(r);if(age<0||m.Length!=v.Length||v.Any(x=>x<0))throw new Exception("Adam values");return age;}
 public static Session ReadSession(string path) {
  var bytes=Payload(path); var s=new Session(); s.hash=Hash(bytes);
  using(var r=new BinaryReader(new MemoryStream(bytes))) {
   if(r.ReadUInt32()!=0x53535446)throw new Exception("session magic");s.version=r.ReadUInt32();if(s.version<4||s.version>6)throw new Exception("session schema");
   var n=r.ReadUInt64();if(n>1024)throw new Exception("task length");s.task=System.Text.Encoding.UTF8.GetString(r.ReadBytes((int)n));
   s.laneSize=(int)r.ReadUInt32();s.countSize=(int)r.ReadUInt32();s.config=new int[8];for(int i=0;i<8;i++)s.config[i]=r.ReadInt32();
   if(r.ReadUInt32()!=20261124)throw new Exception("seed");s.feedback=r.ReadUInt32();s.kind=r.ReadUInt32();s.feature=r.ReadUInt32();
   if(s.feedback!=3||s.kind<2||s.kind>5||s.feature!=(s.kind>=3?2u:1u))throw new Exception("phase mode/layout identity");
   s.constants=new float[10];for(int i=0;i<10;i++)s.constants[i]=r.ReadSingle();r.ReadBytes(48);
   s.gait=r.ReadUInt64();s.offset=r.ReadUInt64();s.starts=r.ReadUInt64();s.initialWeights=r.ReadUInt64();
   s.initialCoefficients=new float[4];for(int i=0;i<4;i++)s.initialCoefficients[i]=r.ReadSingle();
   if(s.version>=5){s.practice=r.ReadUInt32();s.parentPayload=r.ReadUInt64();var pn=r.ReadUInt64();if(pn>1024)throw new Exception("parent identity length");s.parentTask=System.Text.Encoding.UTF8.GetString(r.ReadBytes((int)pn));s.parentUpdate=r.ReadInt32();s.parentActorSteps=r.ReadInt32();s.parentCriticSteps=r.ReadInt32();s.parentInitialWeights=r.ReadUInt64();} if(s.version>=6){s.study=r.ReadUInt32();s.targetSpeed=r.ReadSingle();s.desiredRate=r.ReadSingle();s.material=r.ReadUInt32();s.skipHash=r.ReadUInt64();s.samplingOrigin=r.ReadInt32();} s.identityHash=Hash(bytes.Take(checked((int)r.BaseStream.Position)).ToArray());
   s.update=r.ReadInt32();s.actorSteps=r.ReadInt32();s.criticSteps=r.ReadInt32();s.rng=r.ReadUInt64();s.actor=Floats(r);s.critic=Floats(r);
   var optimizerBegin=(int)r.BaseStream.Position;var a=Adam(r,out s.actorM,out s.actorV,out s.actorAdamConfig);var c=Adam(r,out s.criticM,out s.criticV,out s.criticAdamConfig);s.optimizerBytes=bytes.Skip(optimizerBegin).Take((int)r.BaseStream.Position-optimizerBegin).ToArray();
   if(a!=s.actorSteps||c!=s.criticSteps)throw new Exception("Adam age");s.worlds=r.ReadUInt64();s.nativeWorldBytes=r.ReadBytes(checked((int)s.worlds*s.laneSize));
   s.counters=r.ReadUInt64();if(s.countSize!=72)throw new Exception("unknown count layout");
   for(ulong i=0;i<s.counters;i++){s.attempted+=r.ReadUInt64();s.accepted+=r.ReadUInt64();s.actions+=r.ReadUInt64();s.reliable+=r.ReadUInt64();s.low+=r.ReadUInt64();s.startsCount+=r.ReadUInt32();s.horizons+=r.ReadUInt32();s.clearance+=r.ReadUInt32();s.posture+=r.ReadUInt32();s.left+=r.ReadUInt32();s.right+=r.ReadUInt32();s.measured+=r.ReadUInt32();s.rejected+=r.ReadUInt32();}
   if(r.BaseStream.Position!=r.BaseStream.Length)throw new Exception("trailing session bytes");
  }return s;
 }
}



using System;
using System.IO;
using System.Linq;
public static class BankResumeBudgetCopy {
 public sealed class Result {public int OldTarget,NewTarget;public double OldUsed,NewUsed,Limit;public ulong CheckpointHash;}
 static void Check(bool pass,string message){if(!pass)throw new Exception(message);}
 static ulong Hash(byte[] data){unchecked{ulong h=14695981039346656037UL;foreach(byte b in data){h^=b;h*=1099511628211UL;}return h;}}
 static byte[] Payload(string path){var data=File.ReadAllBytes(path);using(var r=new BinaryReader(new MemoryStream(data))){Check(r.ReadUInt64()==0x31504B434C524443UL,"Unknown wrapper: "+path);var length=r.ReadUInt64();var hash=r.ReadUInt64();Check(length==(ulong)data.Length-24,"Truncated wrapper: "+path);var payload=r.ReadBytes(checked((int)length));Check(Hash(payload)==hash,"Checksum mismatch: "+path);return payload;}}
 static void WriteNew(string path,byte[] payload){using(var stream=new FileStream(path,FileMode.CreateNew,FileAccess.Write,FileShare.None))using(var w=new BinaryWriter(stream)){w.Write(0x31504B434C524443UL);w.Write((ulong)payload.Length);w.Write(Hash(payload));w.Write(payload);w.Flush();stream.Flush(true);}}
 public static Result Inspect(string source,double charged){
  var result=new Result();var latest=Payload(Path.Combine(source,"latest.checkpoint"));var zero=Payload(Path.Combine(source,"checkpoints","session-0.bin"));Check(latest.SequenceEqual(zero),"Latest state is not exact checkpoint zero");result.CheckpointHash=Hash(latest);
  using(var r=new BinaryReader(new MemoryStream(latest))){Check(r.ReadUInt32()==0x52435446&&r.ReadUInt32()==1,"Not FTCR1");var n=r.ReadUInt64();Check(n==64,"Unexpected task identifier");r.ReadBytes((int)n);var expected=new int[]{1024,8,32,9,1024,4,40,64};foreach(var value in expected)Check(r.ReadInt32()==value,"Rate PPO shape differs");r.ReadUInt32();r.ReadUInt32();r.ReadUInt64();r.ReadUInt64();r.ReadUInt64();r.ReadUInt32();r.ReadBytes(16+32+48);Check(r.ReadInt32()==0&&r.ReadInt32()==0&&r.ReadInt32()==0,"Checkpoint already has completed updates/optimizer ages");var count=r.ReadUInt64();Check(count==1353,"Unexpected chooser parameter count");var actor=r.ReadBytes((int)count*4);
   using(var a=new BinaryReader(new MemoryStream(Payload(Path.Combine(source,"actor-0.bin"))))){Check(a.ReadUInt32()==0x48435446&&a.ReadUInt32()==2&&a.ReadUInt32()==1&&a.ReadInt32()==32&&a.ReadInt32()==9&&a.ReadInt32()==40,"Not matching FTCH2 actor");Check(a.ReadUInt64()==count&&a.ReadBytes((int)count*4).SequenceEqual(actor),"Actor zero differs from full-state actor");Check(a.BaseStream.Position==a.BaseStream.Length,"Unexpected actor bytes");}}
  using(var b=new BinaryReader(new MemoryStream(Payload(Path.Combine(source,"budget.checkpoint"))))){Check(b.ReadUInt32()==1&&b.ReadUInt64()==result.CheckpointHash,"Budget/checkpoint binding differs");result.OldTarget=b.ReadInt32();result.Limit=b.ReadDouble();result.OldUsed=b.ReadDouble();Check(b.BaseStream.Position==b.BaseStream.Length,"Unexpected budget bytes");}
  Check(result.OldTarget==64&&result.Limit==1200&&result.OldUsed>=0,"Unexpected original finite budget");Check(!double.IsNaN(charged)&&!double.IsInfinity(charged)&&charged>0,"Invalid prior wall charge");result.NewTarget=63;result.NewUsed=result.OldUsed+charged;Check(result.NewUsed<result.Limit,"Wall budget already spent");return result;
 }
 public static Result Prepare(string source,string destination,double charged){
  Check(!Directory.Exists(destination),"Continuation training destination must not exist");var result=Inspect(source,charged);
  var originalNames=new[]{"latest.checkpoint","checkpoints/session-0.bin","actor-0.bin","budget.checkpoint"};var originals=originalNames.Select(name=>File.ReadAllBytes(Path.Combine(source,name))).ToArray();
  Directory.CreateDirectory(Path.Combine(destination,"checkpoints"));for(int i=0;i<3;++i){var target=Path.Combine(destination,originalNames[i]);File.Copy(Path.Combine(source,originalNames[i]),target,false);Check(File.ReadAllBytes(target).SequenceEqual(originals[i]),"Copy changed checkpoint/actor bytes");}
  byte[] payload;using(var stream=new MemoryStream())using(var w=new BinaryWriter(stream)){w.Write((uint)1);w.Write(result.CheckpointHash);w.Write(result.NewTarget);w.Write(result.Limit);w.Write(result.NewUsed);payload=stream.ToArray();}
  WriteNew(Path.Combine(destination,"budget.checkpoint"),payload);
  for(int i=0;i<originalNames.Length;++i)Check(File.ReadAllBytes(Path.Combine(source,originalNames[i])).SequenceEqual(originals[i]),"Preserved source file changed");
  using(var b=new BinaryReader(new MemoryStream(Payload(Path.Combine(destination,"budget.checkpoint"))))){Check(b.ReadUInt32()==1&&b.ReadUInt64()==result.CheckpointHash&&b.ReadInt32()==63&&b.ReadDouble()==result.Limit&&b.ReadDouble()==result.NewUsed,"Copied budget validation failed");}
  return result;
 }
}

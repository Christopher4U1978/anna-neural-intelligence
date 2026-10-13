import tempfile,unittest,copy
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset
from anna_neural.common import seed_all,source_manifest
from anna_neural.model import ForgeCNN,parameter_count
from anna_neural.data import split_training,training_stats,loader,prepare_data
from anna_neural.training import save_checkpoint,load_checkpoint
from anna_neural.evaluation import evaluate
from anna_neural.optimization import quantize,save_quantized,load_quantized,inference_context
from anna_neural.benchmarking import summarize,measure

class EngineeringTests(unittest.TestCase):
    def setUp(self):seed_all(369,1);self.model=ForgeCNN()
    def test_shapes(self):
        for batch in (1,8,32):self.assertEqual(tuple(self.model(torch.randn(batch,1,8,8)).shape),(batch,10))
    def test_parameter_count(self):self.assertEqual(parameter_count(self.model),38282)
    def test_loss_finite(self):self.assertTrue(torch.isfinite(torch.nn.functional.cross_entropy(self.model(torch.randn(4,1,8,8)),torch.tensor([0,1,2,3]))))
    def test_backward_all_trainable_parameters(self):
        loss=torch.nn.functional.cross_entropy(self.model(torch.randn(4,1,8,8)),torch.tensor([0,1,2,3]));loss.backward()
        for p in self.model.parameters():self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())
    def test_optimizer_changes_weights(self):
        before=self.model.conv1.weight.detach().clone();o=torch.optim.AdamW(self.model.parameters(),lr=.01)
        loss=torch.nn.functional.cross_entropy(self.model(torch.randn(4,1,8,8)),torch.tensor([0,1,2,3]));loss.backward();o.step();self.assertFalse(torch.equal(before,self.model.conv1.weight))
    def test_seed_same_initialization(self):
        seed_all(369,1);other=ForgeCNN()
        for a,b in zip(self.model.parameters(),other.parameters()):self.assertTrue(torch.equal(a,b))
    def test_different_seed_changes_weights(self):seed_all(370,1);self.assertFalse(torch.equal(self.model.conv1.weight,ForgeCNN().conv1.weight))
    def test_repeatable_training_step(self):
        states=[]
        for _ in range(2):
            seed_all(369,1);m=ForgeCNN();o=torch.optim.AdamW(m.parameters(),lr=.01);x=torch.randn(4,1,8,8);y=torch.tensor([0,1,2,3]);torch.nn.functional.cross_entropy(m(x),y).backward();o.step();states.append(m.state_dict())
        for key in states[0]:self.assertTrue(torch.equal(states[0][key],states[1][key]))
    def test_split_disjoint_complete(self):
        labels=np.tile(np.arange(10),20);a,b=split_training(labels,369,.15);self.assertFalse(set(a)&set(b));self.assertEqual(set(a)|set(b),set(range(200)))
    def test_split_reproducible(self):
        labels=np.tile(np.arange(10),20);a,b=split_training(labels,369,.15);c,d=split_training(labels,369,.15);np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
    def test_validation_fraction_bounds(self):
        with self.assertRaises(ValueError):split_training(np.arange(10),369,0)
    def test_normalization_fit_training_only(self):
        train=np.asarray([0.,1.,2.],dtype='float32');mean,std=training_stats(train);self.assertEqual(mean,1.);self.assertAlmostEqual(std,float(np.std(train)))
        test=np.asarray([999.]);self.assertEqual(training_stats(train),(mean,std));self.assertGreater(float(((test-mean)/std)[0]),100)
    def test_constant_normalization_rejected(self):
        with self.assertRaises(ValueError):training_stats(np.zeros((3,1,8,8)))
    def test_loader_batching(self):
        ds=TensorDataset(torch.zeros(10,1,8,8),torch.arange(10));self.assertEqual([len(y) for _,y in loader(ds,4)],[4,4,2])
    def test_loader_shuffle_seed(self):
        ds=TensorDataset(torch.zeros(10,1,8,8),torch.arange(10));a=[y.tolist() for _,y in loader(ds,4,True)];b=[y.tolist() for _,y in loader(ds,4,True)];self.assertEqual(a,b)
    def test_checkpoint_exact_logits_optimizer(self):
        x=torch.randn(3,1,8,8);o=torch.optim.AdamW(self.model.parameters());torch.nn.functional.cross_entropy(self.model(x),torch.arange(3)).backward();o.step();self.model.eval()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt';save_checkpoint(path,self.model,o,{'seed':369},{'mean':1.},1);m,c=load_checkpoint(path);torch.testing.assert_close(m(x),self.model(x),rtol=0,atol=0);o2=torch.optim.AdamW(m.parameters());o2.load_state_dict(c['optimizer_state']);self.assertEqual(len(o2.state),len(o.state))
    def test_evaluation_known_predictions(self):
        class Perfect(torch.nn.Module):
            def forward(self,x):return torch.nn.functional.one_hot(x[:,0,0,0].long(),10).float()*10
        x=torch.zeros(10,1,8,8);x[:,0,0,0]=torch.arange(10);r=evaluate(Perfect(),TensorDataset(x,torch.arange(10)),4)
        self.assertEqual(r['accuracy'],1.);self.assertEqual(r['samples'],10);self.assertEqual(np.trace(r['confusion_matrix']),10);self.assertLess(r['loss'],.001)
    def test_empty_evaluation_rejected(self):
        with self.assertRaises(ValueError):evaluate(self.model,TensorDataset(torch.zeros(0,1,8,8),torch.zeros(0,dtype=torch.long)))
    def test_quantized_shape_and_real_integer_ops(self):
        q=quantize(self.model,TensorDataset(torch.randn(40,1,8,8),torch.zeros(40,dtype=torch.long)),32)
        with inference_context(q):self.assertEqual(tuple(q(torch.randn(3,1,8,8)).shape),(3,10))
        self.assertEqual(q.conv1.weight().dtype,torch.qint8);self.assertEqual(q.fc2.weight().dtype,torch.qint8)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
    def test_quantized_reload(self):
        ds=TensorDataset(torch.randn(40,1,8,8),torch.zeros(40,dtype=torch.long));q=quantize(self.model,ds,32);x=torch.randn(3,1,8,8)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'int8.pt';save_quantized(path,q);r=load_quantized(path,self.model)
            with inference_context(q):torch.testing.assert_close(q(x),r(x),rtol=0,atol=0)
    def test_unsupported_quantization_rejected(self):
        with self.assertRaises(ValueError):quantize(self.model,TensorDataset(torch.zeros(1,1,8,8),torch.zeros(1,dtype=torch.long)),1,'not-a-backend')
    def test_channels_last_same_logits(self):
        m=copy.deepcopy(self.model).to(memory_format=torch.channels_last);x=torch.randn(8,1,8,8);torch.testing.assert_close(self.model(x),m(x.contiguous(memory_format=torch.channels_last)),rtol=1e-5,atol=1e-5)
    def test_benchmark_math(self):
        s=summarize([1.,2.,3.],8);self.assertEqual(s['median_ms'],2.);self.assertEqual(s['throughput_images_s'],4000.);self.assertEqual(s['std_ms'],1.)
    def test_benchmark_bad_inputs(self):
        for times,batch in [([],1),([0.],1),([1.],0),([float('nan')],1),([float('inf')],1)]:
            with self.assertRaises(ValueError):summarize(times,batch)
    def test_benchmark_retains_repeated_samples(self):
        r=measure(self.model.eval(),torch.zeros(1,1,8,8),1,3,2);self.assertEqual(len(r['runs']),2);self.assertEqual(len(r['runs'][0]['raw_ms']),3);self.assertGreater(r['first_forward_ms'],0)
    def test_int8_context_restores_after_exception(self):
        ds=TensorDataset(torch.randn(10,1,8,8),torch.zeros(10,dtype=torch.long));q=quantize(self.model,ds,10)
        with self.assertRaises(RuntimeError):
            with inference_context(q):
                self.assertFalse(torch.are_deterministic_algorithms_enabled());raise RuntimeError('test')
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
    def test_real_schema_pipeline_train_only_statistics(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);rng=np.random.default_rng(369)
            tr=np.column_stack([rng.integers(0,17,(3823,64)),np.arange(3823)%10]);te=np.column_stack([np.full((1797,64),16),np.arange(1797)%10])
            np.savetxt(p/'optdigits.tra',tr,delimiter=',',fmt='%d');np.savetxt(p/'optdigits.tes',te,delimiter=',',fmt='%d')
            with patch('anna_neural.data.ensure_data',return_value=p):ds,meta=prepare_data(p,{'split_seed':369,'val_fraction':.15})
            self.assertEqual(meta['counts'],{'train':3249,'val':574,'test':1797});self.assertAlmostEqual(float(ds['train'].tensors[0].mean()),0,places=5);self.assertGreater(float(ds['test'].tensors[0].mean()),1)
    def test_dataset_tamper_rejected(self):
        from anna_neural.data import ensure_data
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'optdigits.tra').write_text('tampered');(p/'optdigits.tes').write_text('tampered')
            with self.assertRaises(ValueError):ensure_data(p)
    def test_source_manifest(self):
        manifest=source_manifest(Path(__file__).resolve().parents[1]);self.assertIn('src/anna_neural/model.py',manifest);self.assertEqual(len(manifest['src/anna_neural/model.py']),64)

if __name__=='__main__':unittest.main()

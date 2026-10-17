"""Small input-only checks; no model, ROI access, or corpus generation."""
import unittest
import hashlib
from paper_paths import Path, lock_record, rendering_script
import tempfile
from unittest.mock import patch
import numpy as np
from Figure_3.assembly.sft_input_protocol import quantize_sft_input,dequantize_sft_input
from Figure_3.figure_inputs import bounded_sft_inputs,SFT_INPUT_PROTOCOL


class SupervisedQuantizationTests(unittest.TestCase):
    def test_exact_training_quantization_at_both_input_sizes(self):
        for size in (128,256):
            array=np.linspace(0,1,size*size,dtype=np.float32).reshape(1,size,size)
            expected=np.rint(np.clip(array,0,1)*255).astype(np.uint8)
            actual=quantize_sft_input(array)
            self.assertTrue(np.array_equal(actual,expected))
            restored=dequantize_sft_input(actual)
            self.assertEqual(restored.dtype,np.float32)
            self.assertLessEqual(float(np.max(np.abs(restored-array))),.5/255+1e-7)

    def test_stored_bytes_round_trip_exactly(self):
        array=np.arange(256*256,dtype=np.uint8).reshape(1,256,256)
        self.assertTrue(np.array_equal(quantize_sft_input(dequantize_sft_input(array)),array))

    def test_512_intermediate_and_invalid_data_rejected(self):
        for value in (np.zeros((1,512,512)),np.zeros((256,256)),
                      np.full((1,256,256),np.nan),np.full((1,128,128),-1)):
            with self.assertRaises(ValueError):quantize_sft_input(value)
        with self.assertRaises(ValueError):dequantize_sft_input(np.zeros((1,128,128),dtype=np.float32))

    def test_renderer_loads_exact_saved_model_input_bytes(self):
        with tempfile.TemporaryDirectory(prefix='sft-display-test-') as temporary:
            path=Path(temporary)/'inputs.npz'
            arrays={key:np.full(shape,value,dtype=np.uint8) for key,shape,value in
                    (('local_image',(256,256),13),('context_image',(256,256),97),('fine_image',(128,128),201))}
            np.savez_compressed(path,**arrays)
            record={'input_protocol':SFT_INPUT_PROTOCOL,'illustrated_nucleus_id':352082,
                    'outputs':[{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'role':'illustrated_supervised_model_inputs'}]}
            with patch('Figure_3.figure_inputs.locked_input',return_value=path):
                result=bounded_sft_inputs(record)
                for key in arrays:self.assertTrue(np.array_equal(result[key],arrays[key]))
                record['outputs'][0]['sha256']='0'*64
                with self.assertRaises(RuntimeError):bounded_sft_inputs(record)

    def test_renderer_rejects_obsolete_protocol_or_changed_nucleus(self):
        for record in ({'input_protocol':'float_direct_v1','illustrated_nucleus_id':352082},
                       {'input_protocol':SFT_INPUT_PROTOCOL,'illustrated_nucleus_id':1}):
            with self.assertRaises(RuntimeError):bounded_sft_inputs(record)


if __name__=='__main__':unittest.main(verbosity=2)

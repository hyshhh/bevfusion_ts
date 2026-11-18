from ...tensor_quant import QuantDescriptor


class QuantMixin:
    default_quant_desc_input = QuantDescriptor()
    default_quant_desc_weight = QuantDescriptor()

    def _get_quant_desc(self, name, default):
        return default


class QuantInputMixin:
    default_quant_desc_input = QuantDescriptor()

    def _get_quant_desc(self, name, default):
        return default


def pop_quant_desc_in_kwargs(cls, quant_desc_input=None, quant_desc_weight=None, **kw):
    return (quant_desc_input or QuantDescriptor(), quant_desc_weight or QuantDescriptor())

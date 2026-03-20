import re

with open('tests/e2e/test_training_flow.py', 'r') as f:
    content = f.read()

# Make the catch broader (except Exception) and fallback cleaner
# This fixes the TypeError: isinstance() arg 2 must be a type...
content = re.sub(
    r"""        except TypeError:
            # Fallback for the TypeError: isinstance\(\) arg 2 must be a type\.\.\.
            with torch\.no_grad\(\):
                fan = torch\.nn\.init\._calculate_correct_fan\(tensor, mode\)
                gain = torch\.nn\.init\.calculate_gain\(nonlinearity, a\)
                std = gain / math\.sqrt\(fan\)
                bound = math\.sqrt\(3\.0\) \* std
                return tensor\.uniform_\(-float\(bound\), float\(bound\), generator=generator\)""",
    """        except Exception:
            # Fallback for the TypeError: isinstance() arg 2 must be a type...
            with torch.no_grad():
                fan = torch.nn.init._calculate_correct_fan(tensor, mode)
                gain = torch.nn.init.calculate_gain(nonlinearity, a)
                std = float(gain) / math.sqrt(float(fan))
                bound = math.sqrt(3.0) * std
                if generator is not None:
                    return tensor.uniform_(-float(bound), float(bound), generator=generator)
                return tensor.uniform_(-float(bound), float(bound))""",
    content
)

with open('tests/e2e/test_training_flow.py', 'w') as f:
    f.write(content)

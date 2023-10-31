from metaflow import step, FlowSpec, current, card, Parameter
from metaflow.cards import Markdown, Image, Table

class StableDiffusionFlow(FlowSpec):
    prompt = Parameter('prompt',
                    help='The prompt to use for generating an image',
                    default="A minimalistic digital artwork of an abstract geometric pattern with a harmonious color palette")

    @step
    def start(self):
        print("Running diffusion pipeline for: " + self.prompt)
        self.next(self.create_image)

    @card(type='blank')
    @step
    def create_image(self):
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained("OFA-Sys/small-stable-diffusion-v0")
        pipe.to("cpu")

        neg_prompt = "ugly, blurry, poor quality"
        image = pipe(prompt=self.prompt, negative_prompt=neg_prompt).images[0]

        rows = []
        rows.append([Markdown('Prompt: **%s**' % self.prompt)])
        rows.append([Image.from_pil_image(image)])
        current.card.append(Table(rows))
        self.next(self.end)

    @step
    def end(self):
        return

if __name__ == "__main__":
    StableDiffusionFlow()

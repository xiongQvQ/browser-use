"""
GIF generation functionality for agent history visualization.
"""

import base64
import io
import logging
from typing import Optional, Union
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


class GIFGenerator:
    @staticmethod
    def create_history_gif(
        history,
        task: str,
        output_path: str = 'agent_history.gif',
        duration: int = 3000,
        show_goals: bool = True,
        show_task: bool = True,
        show_logo: bool = True,
        font_size: int = 40,
        title_font_size: int = 56,
        goal_font_size: int = 44,
        margin: int = 40,
        line_spacing: float = 1.5,
    ) -> None:
        """Create a GIF from the agent's history with overlaid task and goal text."""
        if not history:
            logger.warning('No history to create GIF from')
            return

        images = []

        # Try to load nicer fonts
        try:
            # Try different font options in order of preference
            font_options = ['Helvetica', 'Arial', 'DejaVuSans', 'Verdana']
            font_loaded = False

            for font_name in font_options:
                try:
                    regular_font = ImageFont.truetype(font_name, font_size)
                    title_font = ImageFont.truetype(font_name, title_font_size)
                    goal_font = ImageFont.truetype(font_name, goal_font_size)
                    font_loaded = True
                    break
                except OSError:
                    continue

            if not font_loaded:
                raise OSError('No preferred fonts found')

        except OSError:
            regular_font = ImageFont.load_default()
            title_font = ImageFont.load_default()
            goal_font = regular_font

        # Load logo if requested
        logo = None
        if show_logo:
            try:
                logo = Image.open('./static/browser-use.png')
                # Resize logo to be small (e.g., 40px height)
                logo_height = 150
                aspect_ratio = logo.width / logo.height
                logo_width = int(logo_height * aspect_ratio)
                logo = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
            except Exception as e:
                logger.warning(f'Could not load logo: {e}')

        # Create task frame if requested
        if show_task and task:
            task_frame = GIFGenerator._create_task_frame(
                task,
                history[0].state.screenshot if history else None,
                title_font,
                regular_font,
                logo,
                line_spacing,
            )
            images.append(task_frame)

        # Process each history item
        for i, item in enumerate(history, 1):
            if not item.state.screenshot:
                continue

            # Convert base64 screenshot to PIL Image
            img_data = base64.b64decode(item.state.screenshot)
            image = Image.open(io.BytesIO(img_data))

            if show_goals and item.model_output:
                image = GIFGenerator._add_overlay_to_image(
                    image=image,
                    step_number=i,
                    goal_text=item.model_output.current_state.next_goal,
                    regular_font=regular_font,
                    title_font=title_font,
                    margin=margin,
                    logo=logo,
                )

            images.append(image)

        if images:
            # Save the GIF
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=0,
                optimize=False,
            )
            logger.info(f'Created history GIF at {output_path}')
        else:
            logger.warning('No images found in history to create GIF')

    @staticmethod
    def _create_task_frame(
        task: str,
        first_screenshot: Optional[str],
        title_font: Union[ImageFont.FreeTypeFont, ImageFont.ImageFont],
        regular_font: Union[ImageFont.FreeTypeFont, ImageFont.ImageFont],
        logo: Optional[Image.Image] = None,
        line_spacing: float = 1.5,
    ) -> Image.Image:
        """Create initial frame showing the task."""
        if not first_screenshot:
            # Create a blank image if no screenshot
            return Image.new('RGB', (800, 600), (0, 0, 0))
            
        img_data = base64.b64decode(first_screenshot)
        template = Image.open(io.BytesIO(img_data))
        image = Image.new('RGB', template.size, (0, 0, 0))
        draw = ImageDraw.Draw(image)

        # Calculate vertical center of image
        center_y = image.height // 2

        # Draw "Task:" title with larger font
        title = 'Task:'
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (image.width - title_width) // 2
        title_y = center_y - 150  # Increased spacing from center

        draw.text(
            (title_x, title_y),
            title,
            font=title_font,
            fill=(255, 255, 255),
        )

        # Draw task text
        margin = 140  # Increased margin
        max_width = image.width - (2 * margin)
        wrapped_text = GIFGenerator._wrap_text(task, regular_font, max_width)

        # Calculate line height with spacing
        # Get font size safely
        font_size = getattr(regular_font, 'size', 16)  # Default to 16 if size not available
        line_height = font_size * line_spacing

        # Split text into lines and draw with custom spacing
        lines = wrapped_text.split('\n')
        total_height = line_height * len(lines)

        # Start position for first line
        text_y = center_y - (total_height / 2) + 50  # Shifted down slightly

        for line in lines:
            # Get line width for centering
            line_bbox = draw.textbbox((0, 0), line, font=regular_font)
            text_x = (image.width - (line_bbox[2] - line_bbox[0])) // 2

            draw.text(
                (text_x, text_y),
                line,
                font=regular_font,
                fill=(255, 255, 255),
            )
            text_y += line_height

        # Add logo if provided
        if logo:
            logo_margin = 20
            logo_x = image.width - logo.width - logo_margin
            image.paste(logo, (logo_x, logo_margin), logo if logo.mode == 'RGBA' else None)

        return image

    @staticmethod
    def _add_overlay_to_image(
        image: Image.Image,
        step_number: int,
        goal_text: str,
        regular_font: Union[ImageFont.FreeTypeFont, ImageFont.ImageFont],
        title_font: Union[ImageFont.FreeTypeFont, ImageFont.ImageFont],
        margin: int,
        logo: Optional[Image.Image] = None,
    ) -> Image.Image:
        """Add step number and goal overlay to an image."""
        image = image.convert('RGBA')
        txt_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(txt_layer)

        # Add step number (bottom left)
        step_text = str(step_number)
        step_bbox = draw.textbbox((0, 0), step_text, font=title_font)
        step_width = step_bbox[2] - step_bbox[0]
        step_height = step_bbox[3] - step_bbox[1]

        # Position step number in bottom left
        x_step = margin + 10  # Slight additional offset from edge
        y_step = image.height - margin - step_height - 10  # Slight offset from bottom

        # Draw rounded rectangle background for step number
        padding = 20  # Increased padding
        step_bg_bbox = (
            x_step - padding,
            y_step - padding,
            x_step + step_width + padding,
            y_step + step_height + padding,
        )
        draw.rounded_rectangle(
            step_bg_bbox,
            radius=15,  # Add rounded corners
            fill=(0, 0, 0, 255),
        )

        # Draw step number
        draw.text(
            (x_step, y_step),
            step_text,
            font=title_font,
            fill=(255, 255, 255, 255),
        )

        # Draw goal text (centered, bottom)
        max_width = image.width - (4 * margin)
        wrapped_goal = GIFGenerator._wrap_text(goal_text, title_font, max_width)
        goal_bbox = draw.multiline_textbbox((0, 0), wrapped_goal, font=title_font)
        goal_width = goal_bbox[2] - goal_bbox[0]
        goal_height = goal_bbox[3] - goal_bbox[1]

        # Center goal text horizontally, place above step number
        x_goal = (image.width - goal_width) // 2
        y_goal = y_step - goal_height - padding * 4  # More space between step and goal

        # Draw rounded rectangle background for goal
        padding_goal = 25  # Increased padding for goal
        goal_bg_bbox = (
            x_goal - padding_goal,
            y_goal - padding_goal,
            x_goal + goal_width + padding_goal,
            y_goal + goal_height + padding_goal,
        )
        draw.rounded_rectangle(
            goal_bg_bbox,
            radius=15,  # Add rounded corners
            fill=(0, 0, 0, 255),
        )

        # Draw goal text
        draw.multiline_text(
            (x_goal, y_goal),
            wrapped_goal,
            font=title_font,
            fill=(255, 255, 255, 255),
            align='center',
        )

        # Add logo if provided
        if logo:
            logo_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
            logo_margin = 20
            logo_x = image.width - logo.width - logo_margin
            logo_layer.paste(logo, (logo_x, logo_margin), logo if logo.mode == 'RGBA' else None)
            txt_layer = Image.alpha_composite(logo_layer, txt_layer)

        # Composite and convert
        result = Image.alpha_composite(image, txt_layer)
        return result.convert('RGB')

    @staticmethod
    def _wrap_text(
        text: str, 
        font: Union[ImageFont.FreeTypeFont, ImageFont.ImageFont], 
        max_width: int
    ) -> str:
        """Wrap text to fit within a given width."""
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            # Try adding the word to the current line
            test_line = ' '.join(current_line + [word])
            bbox = font.getbbox(test_line)
            width = bbox[2] - bbox[0]

            if width <= max_width:
                current_line.append(word)
            else:
                # If the current line is not empty, add it to lines
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        # Add the last line
        if current_line:
            lines.append(' '.join(current_line))

        return '\n'.join(lines) 
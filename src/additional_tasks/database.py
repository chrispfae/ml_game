import ipywidgets as widgets
from IPython.display import display

def create_table():
    # Header
    headers = ['Name', 'Farbe', 'Job', 'Bewegungsart']
    header_row = [widgets.Label(value=h, layout=widgets.Layout(width="150px")) for h in headers]
    header_ui = [widgets.HBox(header_row, layout=widgets.Layout(border="1px solid gray"))]
    
    # Table cells
    rows, cols = 5, 4
    table = [[widgets.Text(placeholder=f"NaN", layout=widgets.Layout(width="150px")) 
              for c in range(cols)] for r in range(rows)]
    row_widgets = [widgets.HBox(row) for row in table]
    
    # Build Table
    table_ui = widgets.VBox(header_ui + row_widgets)
    
    # Search field + button
    search_text = widgets.Text(placeholder="Suchbegriff eingeben")
    search_button = widgets.Button(description="Suchen")
    search_box = widgets.HBox([search_text, search_button])
    
    # Output handling
    out = widgets.Output()
    
    # Display everything
    ui = widgets.VBox([table_ui, search_box, out])
    display(ui)
    
    # --- Example search handler ---
    def on_search_clicked(b):
        query = search_text.value.strip().lower()
        if not query:
            with out:
                out.clear_output()
                display("Bitte Suchtext eingeben")
            return
        
        found_rows = []
        non_found_rows = []
    
        for r, row in enumerate(table):
            row_texts = [cell.value.lower() for cell in row]
            if any(query in text for text in row_texts):
                found_rows.append(row_widgets[r])
            else:
                non_found_rows.append(row_widgets[r])
    
        if found_rows:
            with out:
                out.clear_output()
            # Rebuild table with matches on top
            table_ui.children = tuple(header_ui + found_rows + non_found_rows)
        else:
            with out:
                out.clear_output()
                display("Kein Ergebnis gefunden.")

    search_button.on_click(on_search_clicked)


import ipywidgets as widgets
from IPython.display import display, clear_output

def password():
    richtiges_passwort = ["MachineLearning", "Machine Learning", "Machine-Learning", "machinelearning", "machine learning", "machine-learning", "MACHINELEARNING", "MACHINE LEARNING", "MACHINE-LEARNING"]
    
    # Passwortfeld (versteckt)
    passwort_eingabe = widgets.Password(
        description='Passwort:',
        placeholder='Passwort eingeben'
    )
    
    # Checkbox zum Passwort anzeigen/verstecken
    show_passwort = widgets.Checkbox(
        value=False,
        description='Passwort anzeigen'
    )
    
    # Button zum Bestätigen
    bestaetigen = widgets.Button(description="Bestätigen")
    
    # Ausgabe-Widget für Rückmeldung
    output = widgets.Output()
    
    def toggle_password_visibility(change):
        if change['new']:
            # Sichtbares Textfeld statt Passwortfeld
            visible_pass = widgets.Text(
                value=passwort_eingabe.value,
                description='Passwort:',
                placeholder='Passwort eingeben'
            )
            def on_visible_change(change):
                passwort_eingabe.value = change['new']
            visible_pass.observe(on_visible_change, names='value')
            
            box.children = (visible_pass, show_passwort, bestaetigen, output)
            password_widget_map['field'] = visible_pass
            
        else:
            box.children = (passwort_eingabe, show_passwort, bestaetigen, output)
            password_widget_map['field'] = passwort_eingabe
    
    def pruefe_passwort(b):
        with output:
            clear_output()
            pw = password_widget_map['field'].value
            if pw in richtiges_passwort:
                display(widgets.HTML(value='<span style="color:green; font-weight:bold;">Passwort korrekt! Zugriff gewährt.</span>'))
            else:
                display(widgets.HTML(value='<span style="color:red; font-weight:bold;">Falsches Passwort! Zugriff verweigert.</span>'))
    
    show_passwort.observe(toggle_password_visibility, names='value')
    bestaetigen.on_click(pruefe_passwort)
    
    password_widget_map = {'field': passwort_eingabe}
    box = widgets.VBox([passwort_eingabe, show_passwort, bestaetigen, output])
    display(box)
